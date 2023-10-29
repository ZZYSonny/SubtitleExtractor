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

logging.basicConfig(level=logging.DEBUG)


@dataclass
class ContourConfig:
    white: int
    black: int
    near: int
    kernel: int
    scale: int


@dataclass
class KeyConfig:
    empty: int
    diff: int
    batch_edge: int
    batch_window: int
    margin: int
    contour: ContourConfig
    device: str = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"


KeyConfig1080p1x = KeyConfig(
    200, 1000, 256, 16, 10, ContourConfig(32, 32, 2, 5, 1))
KeyConfig1080p2x = KeyConfig(
    50, 250, 256, 16, 10, ContourConfig(32, 32, 2, 3, 2))
EasyOCRArgs = dict(
    blocklist="~@#$%^&*()_+{}|:\"<>~`[]\\;'/",
    batch_size=2
)

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

@torch.compile(mode="max-autotune")
def subtitle_black_contour(yuv: torch.Tensor, config: ContourConfig):
    y = yuv[:, 0]
    u = yuv[:, 1]
    v = yuv[:, 2]
    grey_mask = torch.logical_and(
        u==128,
        v==128
    )
    white_mask = torch.logical_and(
        y>255-config.white,
        grey_mask
    )
    black_mask = torch.logical_and(
        y<config.black,
        grey_mask
    )
    dtype = torch.float16 if yuv.device.type == "cuda" else torch.float32
    if config.scale == 1:
        white_cnt_scaled = white_mask.to(dtype)
        black_mask_scaled = black_mask
    else:
        white_cnt_scaled = F.avg_pool2d(
            white_mask.to(dtype),
            kernel_size=config.scale,
            divisor_override=1
        )
        black_mask_scaled = F.avg_pool2d(
            black_mask.to(dtype),
            kernel_size=config.scale,
            divisor_override=1
        ) > 0.5
    white_conv = F.avg_pool2d(
        white_cnt_scaled,
        kernel_size=config.kernel,
        padding=config.kernel//2,
        stride=1,
        divisor_override=1
    )
    final = torch.logical_and(
        white_conv.gt(config.near - 0.5),
        black_mask_scaled
    )
    return final


def subtitle_black_contour_single(rgb: torch.Tensor, config: ContourConfig):
    return subtitle_black_contour(rgb.unsqueeze(0), config).squeeze(0)


def subtitle_region(rgb: torch.Tensor):
    return rgb[:, -200:, :]


def subtitle_bound(frame, edge, config: KeyConfig):
    def bound_1d(xs):
        idx = xs.nonzero()
        return max(
            config.contour.scale * idx[0].item()-config.margin,
            0
        ), min(
            config.contour.scale * idx[-1].item()+1+config.margin,
            config.contour.scale * xs.shape[0] - 1
        )

    r1, r2 = bound_1d(edge.int().sum(dim=1))
    c1, c2 = bound_1d(edge.int().sum(dim=0))
    return frame[:, r1:r2, c1:c2]

def get_fps(path):
    video = cv2.VideoCapture(path)
    return video.get(cv2.CAP_PROP_FPS)

def key_frame_generator(path, config: KeyConfig):
    fps=get_fps(path)
    logger = logging.getLogger('KEY')
    stream = torchaudio.io.StreamReader(path)
    stream.add_video_stream(config.batch_edge,
                            decoder="h264_cuvid",
                            hw_accel="cuda:0",
                            decoder_option={
                                "crop": "880x0x0x0"
                            },
                            filter_desc=f"fps={fps}"
                            )
    
    has_start = 0
    start_time = 0.0
    start_frame = torch.empty(0)
    start_edge = torch.empty(0)

    def select_key_frame(cur_time, cur_frame, cur_edge):
        nonlocal has_start, start_time, start_frame, start_edge
        assert (not has_start)
        has_start = True
        start_time = cur_time
        # Move to CPU. start_frame will not be used for computation.
        start_frame = yuv_to_rgb(subtitle_bound(
            subtitle_region(cur_frame),
            cur_edge,
            config
        )).cpu()
        # Clone edge so that in the next batch, previous edge_batch
        # can be deleted.
        start_edge = cur_edge.clone()

    def release_key_frame(cur_time):
        nonlocal has_start
        assert (has_start)
        has_start = False
        return {
            "start": start_time/fps,
            "end": cur_time/fps,
            "frame": start_frame
        }

    past_frames = 0

    logger.info("Decoding video")
    for (yuv_batch, ) in stream.stream():
        logger.info("Computing edges")
        edge_batch = subtitle_black_contour(subtitle_region(yuv_batch), config.contour)
        pixels_batch = edge_batch.int().sum(dim=[1, 2])
        empty_batch_cpu = pixels_batch.lt(config.empty).cpu()

        window_start: int = 0
        while window_start != yuv_batch.shape[0]:
            # Determine current window [cur_start, cur_end]
            window_end: int = min(
                window_start + config.batch_window,
                yuv_batch.shape[0]
            )
            loc = f"{past_frames+window_start} -> {past_frames+window_end}"

            if not has_start:
                logger.debug("="*20)
                logger.debug(
                    f"{loc} Pixs:\n{pixels_batch[window_start:window_end].tolist()}\n")

                non_empty_window_cpu = torch.logical_not(
                    empty_batch_cpu[window_start:window_end])
                i = non_empty_window_cpu.nonzero_static(
                    size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info(f"{loc} Empty:")
                else:
                    window_start = window_start + i
                    logger.info(
                        f"{loc} Empty -> Text at Frame {window_start}")
                    select_key_frame(
                        past_frames+window_start, yuv_batch[window_start], edge_batch[window_start])
            else:
                # Compute diff
                edge_window = edge_batch[window_start:window_end]
                diff_window = torch.logical_xor(
                    edge_window, start_edge).int().sum(dim=[1, 2])
                changed_window_cpu = diff_window.gt(config.diff).cpu()
                empty_window_cpu = empty_batch_cpu[window_start:window_end]
                stop_window_cpu = torch.logical_or(
                    changed_window_cpu, empty_window_cpu)

                logger.debug("="*20)
                logger.debug(
                    f"{loc} Pixs:\n{pixels_batch[window_start:window_end].tolist()}\n")
                logger.debug(f"{loc} Diff:\n{diff_window.tolist()}\n")

                i = stop_window_cpu.nonzero_static(
                    size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info(f"{loc} Text:")
                else:
                    window_start = window_start + i
                    yield release_key_frame(past_frames+window_start)

                    if empty_window_cpu[i].item():
                        logger.info(
                            f"{loc} Text -> Empty  at Frame {window_start}")
                    else:
                        logger.info(
                            f"{loc} Text -> Changed at Frame {window_start}")
                        select_key_frame(
                            past_frames+window_start, yuv_batch[window_start], edge_batch[window_start])

        past_frames += yuv_batch.shape[0]
        logger.info("Decoding video")

    if has_start:
        yield release_key_frame(past_frames-1)


def ocr_text_generator(key_frame_generator, easyocr_args: dict):
    logger = logging.getLogger('KEY')
    reader = easyocr.Reader(['ch_tra', 'en'])
    for key in key_frame_generator:
        image = key['frame'].permute([1, 2, 0]).numpy()
        res = reader.readtext(image, **easyocr_args)
        logger.info(f"[OUT] {res}")
        yield {
            "start": key["start"],
            "end": key["end"],
            "ocrs": res
        }


def subtitle_from_ocr(ocr_result):
    filtered = [
        r for r in ocr_result
        if not r[1].isdigit() and r[0][2][1] - r[0][0][1] > 30
    ]
    return zhconv.convert("\n".join([
        r[1]
        for r in filtered
        if sum(1 if '\u4e00' <= c <= '\u9fff' else 0 for c in r[1]) >= len(r[1])//3
    ]), locale="zh-cn")


def srt_entry_generator(ocrs):
    cnt = 1
    last_start = -1.0
    last_end = -1.0
    last_text = ""
    for i, ocr in enumerate(ocrs):
        cur_start = ocr["start"]
        cur_end = ocr["end"]
        cur_text = subtitle_from_ocr(ocr["ocrs"])
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
                                "crop": "880x0x0x0"
                            }
                            )

    for i, (yuv_batch,) in enumerate(stream.stream()):
        rgb_batch = yuv_to_rgb(yuv_batch)
        torchvision.io.write_png(rgb_batch[0].cpu(), f"debug/img/{i}.png")
        edge = subtitle_black_contour(yuv_batch, config.contour)
        torchvision.io.write_png(bool_to_grey(edge).cpu(), f"debug/img/{i}_.png")


def debug_key(key_frame_generator, config: KeyConfig):
    os.system("rm debug/key/*.png")
    for i, key in enumerate(key_frame_generator):
        frame = key["frame"]
        edge = subtitle_black_contour_single(
            frame, config.contour).to(torch.uint8)
        torchvision.io.write_png(frame, f"debug/key/{i}.png")
        torchvision.io.write_png(
            (edge*255).unsqueeze(0).to(torch.uint8), f"debug/key/{i}_.png")
