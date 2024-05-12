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

LOGLEVEL = os.environ.get('LOGLEVEL', 'ERROR').upper()
logging.basicConfig(level=LOGLEVEL)
reader = easyocr.Reader(['ch_tra'])


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


def bool_to_grey(frames: torch.Tensor):
    return frames.to(torch.uint8).mul(255)


def key_frame_generator(in_video_path, config: FullConfig):
    logger = logging.getLogger('KEY')
    stream = torchaudio.io.StreamReader(in_video_path)

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
    start_debug = None
    start_frame = torch.empty(0, device="cpu")
    start_bound = torch.empty(1, device="cuda:0")

    def select_key_frame(cur_time, cur_cnt, cur_frame, cur_bound):
        nonlocal start_time, start_cnt, start_frame, start_debug, start_bound, has_start
        if has_start and cur_time - start_time < config.key.diff_cd:
            return
        has_start = True
        start_time = cur_time
        start_cnt = cur_cnt
        start_bound = cur_bound
        start_frame = kernels.filter_text_single(cur_frame, cur_bound, config.filter)
        #start_frame = kernels.filter_bounding_single(start_frame, cur_bound)
        start_frame = start_frame.cpu().numpy()
        if LOGLEVEL == "DEBUG": start_debug = cur_frame.cpu()

    def release_key_frame(end_time):
        nonlocal has_start, start_time
        has_start = False
        key = {
            "start": datetime.timedelta(seconds=start_time),
            "end": datetime.timedelta(seconds=end_time),
            "frame": start_frame,
            "debug": start_debug
        }
        start_time = 0.0
        return key
        
    resolution_native = (config.box.width - config.box.left - config.box.right) * (config.box.height - config.box.top - config.box.down)
    threshold_empty = int(config.key.empty_ratio * resolution_native)
    logger.info("Decoding video")
    for (yuv_batch, ) in tqdm(stream.stream(), total=num_batch, desc="Key", position=0):
        pts = yuv_batch.pts

        logger.info("Computing edges")
        yuv = yuv_batch[:]
        bound = kernels.scan_text_boundary(yuv, config.filter)
        cnt_cpu = (bound[:,1] - bound[:,0]).clamp_min(0).sum(dim=[1, 2]).cpu().tolist()
        diff_cpu = (bound - start_bound).abs().sum(dim=[1, 2, 3]).cpu().tolist()
        
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
                    select_key_frame(cur_time, cnt_cpu[i], yuv[i], bound[i])
                    break
        else:
            for i in range(yuv.size(0)):
                if cnt_cpu[i] < threshold_empty:
                    cur_time = pts + i / fps
                    logger.info("Text -> Empty at %s", cur_time)
                    yield release_key_frame(cur_time)
                    break
                if diff_cpu[i] > config.key.diff_ratio * min(start_cnt, cnt_cpu[i]):
                    cur_time = pts + i / fps
                    logger.info("Text -> New at %s", cur_time)
                    yield release_key_frame(cur_time)
                    select_key_frame(cur_time, cnt_cpu[i], yuv[i], bound[i])
                    break

    stream.remove_stream(0)
    if start_time>0:
        yield release_key_frame(num_frame/fps)

def ocr_text_generator(key_frame_generator, config: FullConfig):
    logger = logging.getLogger('OCR')
    logger.info("Loading EasyOCR Model")
    for key in tqdm(key_frame_generator, desc="OCR", position=1):
        if 'ocrs' in key: yield key
        else:
            #img = np.pad(key["frame"], pad_width=32, mode='constant', constant_values=0)
            img = key["frame"]
            res_raw = reader.readtext(img, detail=True, paragraph=False, **config.ocr)
            res_cht = "\n".join(p[1] for p in res_raw)
            res_chs = zhconv.convert(res_cht, locale="zh-cn")
            min_confidence = min((p[2] for p in res_raw), default=0)
            
            logger.info("OCR %f %s",min_confidence, res_chs)
            key["text"] = res_chs
            key["conf"] = min_confidence
            yield key

def debug(key: dict):
    if LOGLEVEL=="DEBUG":
        time = str(key["start"]).replace(":", "_")
        text = str(round(key["conf"], 2)) + "_" + key["text"]
        torchvision.io.write_png(
            torch.from_numpy(key["frame"]).unsqueeze(0),
            f".debug/error/{time}_out_{text}.png"
        )
        if True and key["debug"] is not None:
            torch.save(key["debug"], f".debug/error/{time}.pt")
            torchvision.io.write_png(
                yuv_to_rgb(key["debug"]),
                f".debug/error/{time}_in_{text}.png"
            )


def srt_generator(out_srt_path: str, key_frame_with_text_generator, config: FullConfig):
    entries: list[srt.Subtitle]= []
    
    pbar = tqdm(desc="SRT", position=2)
    for key in key_frame_with_text_generator:
        #debug(key)
        if key["conf"] < config.sub.min_conf:
            debug(key)
        # Generate entry
        elif (len(entries)>0 and key["text"] == entries[-1].content 
        and key["start"] - entries[-1].end < datetime.timedelta(seconds=config.sub.merge_max_sec)):
            entries[-1].end = key["end"]
            #debug(key)
        else:
            start_mod = max(
                key["start"] - datetime.timedelta(seconds=config.sub.fix_delta_sec),
                datetime.timedelta(seconds=0)
            )
            end_mod = max(
                key["end"] - datetime.timedelta(seconds=config.sub.fix_delta_sec),
                datetime.timedelta(seconds=0)
            )
            entries.append(srt.Subtitle(
                index = 0,
                start = start_mod,
                end = end_mod,
                content = key["text"],
            ))
            pbar.update(1)

    with open(out_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(entries))