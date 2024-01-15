import logging
import torch
import torchvision
import torchaudio
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from tqdm import tqdm
import easyocr
from datetime import datetime
import zhconv
import itertools
import os
import cv2
from tqdm import tqdm
import time
import queue
import threading 

LOGLEVEL = os.environ.get('LOGLEVEL', 'ERROR').upper()
logging.basicConfig(level=LOGLEVEL)


@dataclass
class ContourConfig:
    y_tol: int
    uv_tol: int
    black_scale: int
    black_min: int
    white_scale: int
    white_min: int


@dataclass
class KeyConfig:
    empty: float
    diff_tol: float
    batch_edge: int
    batch_window: int
    margin: int
    contour: ContourConfig
    device: str = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"


def yuv_to_rgb(frames):
    frames = frames.to(torch.float)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], -3)
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return rgb


def bool_to_grey(frames: torch.Tensor):
    return frames.to(torch.uint8).mul(255)

@torch.compile
def subtitle_black_contour(yuv: torch.Tensor, config: ContourConfig):
    y = yuv[:, 0]
    u = yuv[:, 1]
    v = yuv[:, 2]
    grey_mask = torch.logical_and(
        torch.logical_and(
            u >= 128 - config.uv_tol,
            u <= 128 + config.uv_tol
        ),
        torch.logical_and(
            v >= 128 - config.uv_tol,
            v <= 128 + config.uv_tol
        )
    )
    white_mask = torch.logical_and(
        y > 255-config.y_tol,
        grey_mask
    )
    black_mask = torch.logical_and(
        y < config.y_tol,
        grey_mask
    )
    white_mask_scaled = torch.sum(
        white_mask.reshape([
            y.size(0), 
            y.size(1)//config.white_scale, config.white_scale, 
            y.size(2)//config.white_scale, config.white_scale, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.white_min)
    black_mask_scaled = torch.sum(
        black_mask.reshape([
            y.size(0), 
            y.size(1)//config.black_scale, config.black_scale, 
            y.size(2)//config.black_scale, config.black_scale, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.black_min)
    
    if config.white_scale < config.black_scale:
        black_mask_scaled = black_mask_scaled.repeat_interleave(
            config.black_scale//config.white_scale, 
            dim=1
        ).repeat_interleave(
            config.black_scale//config.white_scale, 
            dim=2
        )
    elif config.white_scale > config.black_scale:
        white_mask_scaled = white_mask_scaled.repeat_interleave(
            config.white_scale//config.black_scale, 
            dim=1
        ).repeat_interleave(
            config.white_scale//config.black_scale, 
            dim=2
        )

    final = torch.logical_and(
        white_mask_scaled,
        black_mask_scaled
    )
    return final

@torch.compile
def mask_non_text_area(frame: torch.Tensor, edge: torch.Tensor, config: KeyConfig):
    scale = min(config.contour.black_scale, config.contour.white_scale)
    not_edge_patched = torch.logical_not(edge).repeat_interleave(
        scale, dim=0
    ).repeat_interleave(
        scale, dim=1
    )
    frame[not_edge_patched] = 0
    return frame

def subtitle_black_contour_single(rgb: torch.Tensor, config: ContourConfig):
    return subtitle_black_contour(rgb.unsqueeze(0), config).squeeze(0)

def subtitle_region(rgb: torch.Tensor):
    return rgb[:, -192:, :]

def bounding_box(frame, edge, config: KeyConfig):
    scale = min(config.contour.black_scale, config.contour.white_scale)
    # Crop the bounding box
    def bound_1d(xs):
        idx = xs.nonzero()
        r = max(
            idx[0].item()-config.margin,
            0
        )
        c = min(
            idx[-1].item()+1+config.margin,
            xs.shape[0] - 1
        )
        return r,c
    r1, r2 = bound_1d(edge.sum(dim=1, dtype=torch.int32))
    c1, c2 = bound_1d(edge.sum(dim=0, dtype=torch.int32))
    frame_box = frame[..., scale*r1:scale*r2, scale*c1:scale*c2]
    return frame_box

def async_iterable(xs, length):
    q = queue.Queue(2)

    def decode_worker():
        for x in xs:
            q.put(x)
        q.task_done()

    threading.Thread(target=decode_worker, daemon=True).start()
    for i in range(length):
        yield q.get()
        

def key_ocr_generator(path, config: KeyConfig, ocr_args: dict):
    logger = logging.getLogger('KEY')
    stream = torchaudio.io.StreamReader(path)
    reader = easyocr.Reader(['ch_tra', 'en'])

    info = stream.get_src_stream_info(0)
    num_frame = stream.get_src_stream_info(0).num_frames
    num_batch = int((num_frame + config.batch_edge - 1) // config.batch_edge)
    assert (info.codec == "h264")
    assert (info.format == "yuv420p")

    fps = info.frame_rate
    stream.add_video_stream(config.batch_edge,
                            decoder="h264_cuvid",
                            hw_accel="cuda:0",
                            decoder_option={
                                "crop": "888x0x0x0"
                            },
                            #filter_desc=f"fps={fps}"
                            )

    has_start = 0
    start_time = 0
    start_frame = torch.empty(0)
    start_edge = torch.empty(0)
    start_pix = torch.empty(1)
    start_ocr = []

    def select_key_frame(cur_time, cur_frame, cur_edge):
        nonlocal has_start, start_time, start_frame, start_edge, start_pix, start_ocr
        assert (not has_start)
        has_start = True
        start_time = cur_time
        start_pix = cur_edge.sum()
        # Move to CPU. start_frame will not be used for computation.
        start_subtitle_area = subtitle_region(cur_frame)
        start_grey = mask_non_text_area(start_subtitle_area[0], cur_edge, config)
        start_grey_bounded = bounding_box(start_grey, cur_edge, config)
        start_ocr = reader.recognize(start_grey_bounded.cpu().numpy())

        # Clone edge so that in the next batch, previous edge_batch
        # can be deleted.
        start_edge = cur_edge

    def release_key_frame(cur_time):
        nonlocal has_start
        assert (has_start)
        has_start = False
        return {
            "start": start_time/fps,
            "end": cur_time/fps,
            "ocrs": start_ocr
        }

    past_frames = 0

    logger.info("Decoding video")
    for (yuv_batch, ) in tqdm(async_iterable(stream.stream(), num_batch), total=num_batch, desc="Key"):
        logger.info("Computing edges")
        edge_batch = subtitle_black_contour(
            subtitle_region(yuv_batch), config.contour)
        pixels_batch = edge_batch.int().sum(dim=[1, 2])
        empty_batch_cpu = pixels_batch.lt(edge_batch[0].numel() * config.empty).cpu()

        window_start: int = 0
        while window_start != yuv_batch.shape[0]:
            # Determine current window [cur_start, cur_end]
            window_end: int = min(
                window_start + config.batch_window,
                yuv_batch.shape[0]
            )
            loc = f"{past_frames+window_start} -> {past_frames+window_end}"

            if not has_start:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("="*20)
                    logger.debug(
                        "%s Pixs:\n%s\n", loc, pixels_batch[window_start:window_end].tolist())

                non_empty_window_cpu = torch.logical_not(
                    empty_batch_cpu[window_start:window_end])
                i = non_empty_window_cpu.nonzero_static(
                    size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info("%s Empty:", loc)
                else:
                    window_start = window_start + i
                    logger.info("%s Empty -> New Text at Frame %s",
                                loc, past_frames+window_start)
                    select_key_frame(
                        past_frames+window_start, yuv_batch[window_start], edge_batch[window_start])
            else:
                # Compute diff
                edge_window = edge_batch[window_start:window_end]
                diff_window = torch.logical_xor(
                    edge_window, start_edge).int().sum(dim=[1, 2])
                diff_thres = torch.min(
                    start_pix,
                    pixels_batch[window_start:window_end]
                ).mul(config.diff_tol).int()
                changed_window_cpu = diff_window.gt(diff_thres).cpu()
                empty_window_cpu = empty_batch_cpu[window_start:window_end]
                stop_window_cpu = torch.logical_or(
                    changed_window_cpu, empty_window_cpu)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("="*20)
                    logger.debug("%s Pixs:%s\n", loc,
                                 pixels_batch[window_start:window_end].tolist())
                    logger.debug("%s Diff:%s\n", loc, diff_window.tolist())

                i = stop_window_cpu.nonzero_static(
                    size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info("%s Text:", loc)
                else:
                    window_start = window_start + i
                    yield release_key_frame(past_frames+window_start)

                    if empty_window_cpu[i].item():
                        logger.info("%s Text -> Empty  at Frame %s",
                                    loc, past_frames+window_start)
                    else:
                        logger.info("%s Text -> New Text at Frame %s",
                                    loc, past_frames+window_start)
                        select_key_frame(
                            past_frames+window_start, yuv_batch[window_start], edge_batch[window_start])

        past_frames += yuv_batch.shape[0]
        logger.info("Decoding video")

    stream.remove_stream(0)
    if has_start:
        yield release_key_frame(past_frames-1)

def text_for_subtitle(ocr_result):
    return zhconv.convert("\n".join([
        r[1]
        for r in ocr_result
        if not r[1].isdigit() and r[0][2][1] - r[0][0][1] > 30
        if sum(1 if '\u4e00' <= c <= '\u9fff' else 0 for c in r[1]) >= max(1,len(r[1])//4)
    ]), locale="zh-cn")


def srt_entry_generator(ocrs):
    cnt = 1
    last_start = -1.0
    last_end = -1.0
    last_text = ""
    for i, ocr in enumerate(ocrs):
        cur_start = ocr["start"]
        cur_end = ocr["end"]
        cur_text = text_for_subtitle(ocr["ocrs"])
        if last_text == cur_text and cur_start - last_end < 0.1:
            last_end = cur_end 
        else:
            if len(last_text) > 0:
                time1 = datetime.utcfromtimestamp(
                    last_start).strftime('%H:%M:%S,%f')[:-3]
                time2 = datetime.utcfromtimestamp(
                    last_end).strftime('%H:%M:%S,%f')[:-3]
                yield "\n".join([
                    f"{cnt}",
                    f"{time1} --> {time2}",
                    last_text
                ])
                cnt += 1
            last_start = cur_start
            last_end = cur_end
            last_text = cur_text
    time1 = datetime.utcfromtimestamp(last_start).strftime('%H:%M:%S,%f')[:-3]
    time2 = datetime.utcfromtimestamp(last_end).strftime('%H:%M:%S,%f')[:-3]
    yield "\n".join([
        f"{cnt}",
        f"{time1} --> {time2}",
        last_text
    ])


def debug_contour(path, config: KeyConfig):
    os.system("rm debug/img/*.png")
    stream = torchaudio.io.StreamReader(path)
    stream.add_video_stream(1,
                            decoder="h264_cuvid",
                            hw_accel="cuda:0",
                            decoder_option={
                                "crop": "888x0x0x0"
                            }
                            )

    for i, (yuv_batch,) in enumerate(stream.stream()):
        #if i!=227: continue
        yuv_batch = subtitle_region(yuv_batch)
        rgb_batch = yuv_to_rgb(yuv_batch)
        edge_batch = subtitle_black_contour(yuv_batch, config.contour)

        torchvision.io.write_png(rgb_batch[0].cpu(), f"debug/img/{i}.png")

        if edge_batch.sum().item() > edge_batch[0].numel() * config.empty:
            rgb_cut = post_process(
                rgb_batch[0],
                edge_batch[0],
                config
            )
            torchvision.io.write_png(bool_to_grey(
                edge_batch).cpu(), f"debug/img/{i}_.png")
            torchvision.io.write_png(rgb_cut, f"debug/img/{i}__.png")
