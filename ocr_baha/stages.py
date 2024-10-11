import os
import logging
from dataclasses import dataclass
import datetime


from tqdm import tqdm
import numpy as np
import torch
import torchvision
import torchaudio
import easyocr
import zhconv
import srt

from . import kernels
from .kernels import FilterConfig

LOGLEVEL = os.environ.get("LOGLEVEL", "ERROR").upper()
logging.basicConfig(level=LOGLEVEL)
reader = easyocr.Reader(["ch_tra"])


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
    empty_ratio: float
    diff_ratio: float
    diff_cd: float


@dataclass
class SubsConfig:
    min_conf: float
    fix_delta_sec: float
    merge_max_sec: float


@dataclass
class FullConfig:
    exe: ExecConfig
    key: KeyConfig
    box: CropConfig
    filter: FilterConfig
    ocr: dict
    sub: SubsConfig


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


def single_mask(yuv):
    y, u, v = yuv
    a = 19
    b = 1
    return (
        (y >= 255 - a)
        & (u >= 128 - b)
        & (u <= 128 + b)
        & (v >= 128 - b)
        & (v <= 128 + b)
    )


def combine_mask(last, yuv):
    return last & single_mask(yuv)


def bool_to_grey(frames: torch.Tensor):
    return frames.to(torch.uint8).mul(255)


def key_frame_generator(in_video_path, config: FullConfig):
    logger = logging.getLogger("KEY")
    stream = torchaudio.io.StreamReader(in_video_path)

    num_frame = stream.get_src_stream_info(0).num_frames
    info = stream.get_src_stream_info(0)
    fps = info.frame_rate
    assert info.codec == "h264"
    assert info.format == "yuv420p"
    resolution_native = (config.box.width - config.box.left - config.box.right) * (
        config.box.height - config.box.top - config.box.down
    )
    threshold_empty = int(config.key.empty_ratio * resolution_native)

    stream.add_video_stream(
        config.exe.batch,
        decoder="h264_cuvid",
        hw_accel="cuda:0",
        decoder_option={
            "crop": f"{config.box.top}x{config.box.down}x{config.box.left}x{config.box.left}x{config.box.right}"
        },
        # filter_desc=f"fps={fps}"
    )
    start_time = None
    start_frame = None
    start_mask = None
    start_cnt = 0

    def select_key_frame(cur_time, cur_frame):
        nonlocal start_time, start_cnt, start_frame, start_mask
        assert start_time is None
        mask = single_mask(cur_frame)
        cnt = mask.sum(dtype=torch.int).item()
        if cnt > threshold_empty:
            start_time = cur_time
            start_frame = cur_frame
            start_mask = mask
            start_cnt = cnt

    def release_key_frame(end_time):
        nonlocal start_time, start_frame, start_mask, start_cnt
        assert start_time is not None

        # processed = torch.where(start_mask, start_frame[0], 0)
        key = None
        if end_time - start_time > config.key.diff_cd:
            key = {
                "start": datetime.timedelta(seconds=start_time),
                "end": datetime.timedelta(seconds=end_time),
                "frame": start_frame[0].cpu().numpy(),
                "debug": start_frame.cpu().numpy(),
            }
        start_time = None
        start_frame = None
        start_mask = None
        start_cnt = 0

        if key is not None:
            yield key

    logger.info("Decoding video")
    for (yuv,) in tqdm(stream.stream(), total=num_frame, desc="Key", position=0):
        if yuv is None:
            continue
        pts = yuv.pts

        if start_time is None:
            select_key_frame(pts, yuv[0])
        else:
            cur_mask = combine_mask(start_mask, yuv[0])
            cur_cnt = cur_mask.sum().item()
            if cur_cnt < threshold_empty:
                yield from release_key_frame(pts)

    stream.remove_stream(0)
    if start_time is not None:
        yield from release_key_frame(num_frame / fps)


def ocr_text_generator(key_frame_generator, config: FullConfig):
    logger = logging.getLogger("OCR")
    logger.info("Loading EasyOCR Model")
    for key in tqdm(key_frame_generator, desc="OCR", position=1):
        if "ocrs" in key:
            yield key
        else:
            # img = np.pad(key["frame"], pad_width=32, mode='constant', constant_values=0)
            img = key["frame"]
            if img is None:
                return None
            res_raw = reader.readtext(img, detail=True, paragraph=False, **config.ocr)
            res_cht = "\n".join(p[1] for p in res_raw)
            res_chs = zhconv.convert(res_cht, locale="zh-cn")
            min_confidence = min((p[2] for p in res_raw), default=0)

            logger.info("OCR %f %s", min_confidence, res_chs)
            key["text"] = res_chs
            key["conf"] = min_confidence
            yield key


def debug(key: dict):
    if LOGLEVEL == "DEBUG":
        time = str(key["start"]).replace(":", "_")
        text = str(round(key["conf"], 2)) + "_" + key["text"]
        torchvision.io.write_png(
            torch.from_numpy(key["frame"]).unsqueeze(0),
            f".debug/error/{time}_out_{text}.png",
        )
        if True and key["debug"] is not None:
            torch.save(key["debug"], f".debug/error/{time}.pt")
            torchvision.io.write_png(
                torch.from_numpy(key["debug"][0]).unsqueeze(0),
                f".debug/error/{time}_in_{text}.png",
            )


def srt_generator(out_srt_path: str, key_frame_with_text_generator, config: FullConfig):
    entries: list[srt.Subtitle] = []

    pbar = tqdm(desc="SRT", position=2)
    for key in key_frame_with_text_generator:
        # debug(key)
        if key["conf"] < config.sub.min_conf:
            debug(key)
        # Generate entry
        elif (
            len(entries) > 0
            and key["text"] == entries[-1].content
            and key["start"] - entries[-1].end
            < datetime.timedelta(seconds=config.sub.merge_max_sec)
        ):
            entries[-1].end = key["end"]
            debug(key)
        else:
            start_mod = max(
                key["start"] + datetime.timedelta(seconds=config.sub.fix_delta_sec),
                datetime.timedelta(seconds=0),
            )
            end_mod = max(
                key["end"] + datetime.timedelta(seconds=config.sub.fix_delta_sec),
                datetime.timedelta(seconds=0),
            )
            entries.append(
                srt.Subtitle(
                    index=0,
                    start=start_mod,
                    end=end_mod,
                    content=key["text"],
                )
            )
            pbar.update(1)

    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(entries))
