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
import numpy as np

LOGLEVEL = os.environ.get('LOGLEVEL', 'ERROR').upper()
logging.basicConfig(level=LOGLEVEL)
reader = easyocr.Reader(['ch_tra'])


@dataclass
class ContourConfig:
    y_white_tol: int
    y_black_tol: int
    uv_tol: int
    black_x_scale: int
    black_y_scale: int
    black_min: int
    white_x_scale: int
    white_y_scale: int
    white_min: int

@dataclass
class CropConfig:
    top: int
    down: int
    left: int
    right: int
    width: int
    height: int

@dataclass
class ExecConfig:
    batch: int
    device: str = "cuda"

@dataclass
class KeyConfig:
    empty: float
    diff_tol: float

@dataclass
class SubsConfig:
    exe: ExecConfig
    key: KeyConfig
    box: CropConfig
    contour: ContourConfig
    ocr: dict


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
        y > 255-config.y_white_tol,
        grey_mask
    )
    black_mask = torch.logical_and(
        y < config.y_black_tol,
        grey_mask
    )
    white_mask_scaled = torch.sum(
        white_mask.reshape([
            y.size(0), 
            y.size(1)//config.white_x_scale, config.white_x_scale, 
            y.size(2)//config.white_y_scale, config.white_y_scale, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.white_min)
    black_mask_scaled = torch.sum(
        black_mask.reshape([
            y.size(0), 
            y.size(1)//config.black_x_scale, config.black_x_scale, 
            y.size(2)//config.black_y_scale, config.black_y_scale, 
        ]),
        dim=[2,4],
        dtype=torch.uint8
    ).greater_equal(config.black_min)
    
    if config.white_x_scale < config.black_x_scale:
        black_mask_scaled = black_mask_scaled.repeat_interleave(
            config.black_x_scale//config.white_x_scale, 
            dim=1
        )

    if config.white_y_scale < config.black_y_scale:
        black_mask_scaled = black_mask_scaled.repeat_interleave(
            config.black_y_scale//config.white_y_scale, 
            dim=2
        )

    if config.white_x_scale > config.black_x_scale:
        white_mask_scaled = white_mask_scaled.repeat_interleave(
            config.white_x_scale//config.black_x_scale, 
            dim=1
        )

    if config.white_y_scale > config.black_y_scale:
        white_mask_scaled = white_mask_scaled.repeat_interleave(
            config.white_y_scale//config.black_y_scale, 
            dim=2
        )

    final = torch.logical_and(
        white_mask_scaled,
        black_mask_scaled
    )
    return final

@torch.compile
def mask_non_text_area(frame: torch.Tensor, edge: torch.Tensor, config: ContourConfig):
    scale_x = min(config.black_x_scale, config.white_x_scale)
    scale_y = min(config.black_y_scale, config.white_y_scale)
    not_edge_patched = torch.logical_not(edge).repeat_interleave(
        scale_x, dim=0
    ).repeat_interleave(
        scale_y, dim=1
    )
    frame[not_edge_patched] = 0
    return frame

def subtitle_black_contour_single(rgb: torch.Tensor, config: ContourConfig):
    return subtitle_black_contour(rgb.unsqueeze(0), config).squeeze(0)

#def subtitle_region(rgb: torch.Tensor):
#    return rgb[:, -192:, :]

def bounding_box(frame, edge, config: ContourConfig):
    scale_x = min(config.black_x_scale, config.white_x_scale)
    scale_y = min(config.black_y_scale, config.white_y_scale)
    # Crop the bounding box
    def bound_1d(xs):
        idx = xs.nonzero()
        r = idx[0].item()
        c = idx[-1].item()+1
        return r,c
    r1, r2 = bound_1d(edge.sum(dim=1, dtype=torch.int32))
    c1, c2 = bound_1d(edge.sum(dim=0, dtype=torch.int32))
    frame_box = frame[..., scale_x*r1:scale_x*r2, scale_y*c1:scale_y*c2]
    return frame_box

def async_iterable(xs, limit=2):
    def async_generator():
        q = queue.Queue(limit)

        def decode_worker():
            for x in xs:
                q.put(x)
            q.put(None)
            q.task_done()
        threading.Thread(target=decode_worker, daemon=True).start()
        while True:
            x = q.get()
            if x is None: break
            else: yield x

    if isinstance(xs, list): return xs
    else: return async_generator()

        

def key_frame_generator(path, config: SubsConfig):
    logger = logging.getLogger('KEY')
    stream = torchaudio.io.StreamReader(path)

    info = stream.get_src_stream_info(0)
    num_frame = stream.get_src_stream_info(0).num_frames
    num_batch = int((num_frame + config.exe.batch - 1) // config.exe.batch)
    assert (info.codec == "h264")
    assert (info.format == "yuv420p")

    fps = info.frame_rate
    stream.add_video_stream(config.exe.batch,
                            decoder="h264_cuvid",
                            hw_accel="cuda:0",
                            decoder_option={
                                "crop": f"{config.box.top}x{config.box.down}x{config.box.left}x{config.box.left}x{config.box.right}"
                            },
                            #filter_desc=f"fps={fps}"
                            )
    has_start = False
    start_time = 0.0
    start_cnt = 0
    start_frame = torch.empty(0, device="cpu")
    start_edge = torch.empty(1, device="cuda:0")

    def select_key_frame(cur_time, cur_cnt, cur_frame, cur_edge):
        nonlocal start_time, start_cnt, start_frame, start_edge, has_start
        has_start = True
        start_time = cur_time
        start_cnt = cur_cnt
        start_edge = cur_edge
        if False:
            torchvision.io.write_png(yuv_to_rgb(cur_frame).cpu(), f"debug/img/{start_time}_in.png")
            torchvision.io.write_png(cur_frame[0][None,:].cpu(), f"debug/img/{start_time}_y.png")
            torchvision.io.write_png(cur_frame[1][None,:].cpu(), f"debug/img/{start_time}_u.png")
            torchvision.io.write_png(cur_frame[2][None,:].cpu(), f"debug/img/{start_time}_v.png")
            torch.save(cur_frame, f"debug/img/{start_time}.pt")
            #torchvision.io.write_png(torch.from_numpy(start_frame)[None,:], f"debug/img/{start_time}_out.png")
        start_grey = mask_non_text_area(cur_frame[0], cur_edge, config.contour)
        start_frame = bounding_box(start_grey, cur_edge, config.contour).cpu().numpy()

    def release_key_frame(end_time):
        nonlocal has_start, start_time
        has_start = False
        key = {
            "start": start_time,
            "end": end_time,
            "frame": start_frame
        }
        start_time = 0.0
        return key
        
    resolution_native = (config.box.width - config.box.left - config.box.right) * (config.box.height - config.box.top - config.box.down)
    resolution_edge = resolution_native / min(config.contour.black_x_scale, config.contour.white_x_scale) / min(config.contour.black_y_scale, config.contour.white_y_scale)
    threshold_empty = int(config.key.empty*resolution_edge)
    logger.info("Decoding video")
    for (yuv_batch, ) in tqdm(stream.stream(), total=num_batch, desc="Key", position=0):
        pts = yuv_batch.pts

        logger.info("Computing edges")
        yuv = yuv_batch[:]
        edge = subtitle_black_contour(yuv, config.contour)
        cnt = edge.int().sum(dim=[1, 2])
        diff = torch.logical_xor(edge, start_edge).int().sum(dim=[1, 2])

        cnt_cpu = cnt.cpu().tolist()
        diff_cpu = diff.cpu().tolist()
        
        if logger.isEnabledFor(logging.DEBUG):
            time = [
                pts + i / fps
                for i in range(yuv.size(0))
            ]
            logger.debug("="*20)
            logger.debug("Time:%s", time)
            logger.debug("Pixs:%s", cnt_cpu)
            logger.debug("Diff:%s", diff_cpu)

        # Assume in a batch, the state changes at most once.
        if not has_start:
            for i in range(yuv.size(0)):
                if cnt_cpu[i] > threshold_empty:
                    cur_time = pts + i / fps
                    logger.info("Empty -> Text at %s", cur_time)
                    select_key_frame(cur_time, cnt[i], yuv[i], edge[i])
                    break
        else:
            for i in range(yuv.size(0)):
                if cnt_cpu[i] < threshold_empty:
                    cur_time = pts + i / fps
                    logger.info("Text -> Empty at %s", cur_time)
                    yield release_key_frame(cur_time)
                    break
                if diff_cpu[i] > config.key.diff_tol * min(start_cnt, cnt[i]):
                    cur_time = pts + i / fps
                    logger.info("Text -> New at %s", cur_time)
                    yield release_key_frame(cur_time)
                    select_key_frame(cur_time, cnt[i], yuv[i], edge[i])
                    break

    stream.remove_stream(0)
    if start_time>0:
        yield release_key_frame(num_frame/fps)

def ocr_text_generator(key_frame_generator, config: SubsConfig):
    logger = logging.getLogger('OCR')
    logger.info("Loading EasyOCR Model")
    for key in tqdm(key_frame_generator, desc="OCR", position=1):
        if 'ocrs' in key: yield key
        else:
            img = np.pad(key["frame"], pad_width=16, mode='constant', constant_values=0)
            res_raw = reader.readtext(img, detail=True, paragraph=False, **config.ocr)
            res_cht = "\n".join(p[1] for p in res_raw)
            min_confidence = min((p[2] for p in res_raw), default=0)
            if min_confidence >= 0.2:
                res_chs = zhconv.convert(res_cht, locale="zh-cn")
                logger.info("%s", res_chs)
                yield {
                    "start": key["start"],
                    "end": key["end"],
                    "ocrs": res_chs
                }
            elif LOGLEVEL=="DEBUG":
                torchvision.io.write_png(
                    torch.from_numpy(key["frame"]).unsqueeze(0),
                    f"debug/error/{key['start']:0>6.1f}_{min_confidence:.2f}_{res_cht}.png"
                )


def srt_entry_generator(ocrs):
    cnt = 1
    last_start = -1.0
    last_end = -1.0
    last_text = ""
    for i, ocr in tqdm(enumerate(ocrs), desc="SRT", position=2):
        cur_start = ocr["start"]
        cur_end = ocr["end"]
        cur_text = ocr["ocrs"]
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


#def debug_contour(path, config: SubsConfig):
#    os.system("rm debug/img/*.png")
#    stream = torchaudio.io.StreamReader(path)
#    stream.add_video_stream(1,
#                            decoder="h264_cuvid",
#                            hw_accel="cuda:0",
#                            decoder_option={
#                                "crop": "888x0x0x0"
#                            }
#                            )
#
#    for i, (yuv_batch,) in enumerate(stream.stream()):
#        #if i!=227: continue
#        yuv_batch = subtitle_region(yuv_batch)
#        rgb_batch = yuv_to_rgb(yuv_batch)
#        edge_batch = subtitle_black_contour(yuv_batch, config.contour)
#
#        torchvision.io.write_png(rgb_batch[0].cpu(), f"debug/img/{i}.png")
#
#        if edge_batch.sum().item() > edge_batch[0].numel() * config.empty:
#            rgb_cut = post_process(
#                rgb_batch[0],
#                edge_batch[0],
#                config
#            )
#            torchvision.io.write_png(bool_to_grey(
#                edge_batch).cpu(), f"debug/img/{i}_.png")
#            torchvision.io.write_png(rgb_cut, f"debug/img/{i}__.png")
#