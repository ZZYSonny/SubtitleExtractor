import logging
import torch
import torchvision
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from tqdm import tqdm
import easyocr
from datetime import datetime
import zhconv

logging.basicConfig(level=logging.INFO)

@dataclass
class ContourConfig:
    white: int
    black: int
    near: int
    kernel: int

@dataclass
class KeyConfig:
    empty: int = 200
    diff: int = 1000
    batch: int = 128*3
    contour = ContourConfig(8,8,2,5)
    device: str = "cuda"

@dataclass
class OCRConfig:
    batch: int = 2
    min_conf: float = 0.3


@torch.compile(mode="max-autotune")
def subtitle_black_contour(rgb: torch.Tensor, config: ContourConfig):
    channel_dim = 1
    r = rgb.select(channel_dim, 0).unsqueeze(channel_dim)
    mask = rgb.eq(r).all(dim=channel_dim)
    white = torch.logical_and(
        rgb.gt(255 - config.white).all(dim=channel_dim),
        mask
    )
    black = torch.logical_and(
        rgb.lt(config.black).all(dim=channel_dim),
        mask
    )
    white_conv = F.avg_pool2d(
        white.float(),
        kernel_size=config.kernel, 
        padding=config.kernel//2, 
        stride=1,
        divisor_override=1
    )
    final = torch.logical_and(
        white_conv.gt(config.near - 0.5),
        black
    )
    return final

def subtitle_black_contour_single(rgb: torch.Tensor, config: ContourConfig):
    return subtitle_black_contour(rgb.unsqueeze(0), config).squeeze(0)

def subtitle_region(rgb: torch.Tensor):
    return rgb[:, -200:, :]

def subtitle_diff_sum(contours, reference):
    return torch.logical_xor(contours, reference).int().sum(dim=[1,2])


def batcher(iter, batch_size):
    global config
    bufs = []
    for x in iter:
        bufs.append(x)
        if len(bufs) == batch_size:
            yield bufs
            bufs = []
    if len(bufs) > 0:
        yield bufs


def key_frame_generator(path, config: KeyConfig = KeyConfig()):
    logger = logging.getLogger('KEY')
    reader = torchvision.io.VideoReader(path, "video", num_threads=4)

    start_time = 0.0
    start_frame = None
    start_contour = None

    for video_frames in batcher(reader, config.batch):
        frame_batch = torch.stack([
            subtitle_region(frame["data"])
            for frame in video_frames
        ]).to(config.device)
        edge_batch = subtitle_black_contour(frame_batch, config.contour)
        edge_sum = edge_batch.int().sum(dim=[1,2]).tolist()

        if start_contour is not None:
            diffs_sum = subtitle_diff_sum(edge_batch, start_contour).tolist()
        else:
            diffs_sum = None
        
        for i in range(frame_batch.shape[0]):
            cur_time = video_frames[i]["pts"]
            if start_contour is None:
                logger.info(f"[STATS] i={cur_time} pix={edge_sum[i]}")
                # no starting frame
                if edge_sum[i] > config.empty:
                    # current frame has text
                    logger.info(f"[EVENT]::EMPTY->TEXT {cur_time}")
                    start_time = cur_time
                    start_frame = frame_batch[i]
                    start_contour = edge_batch[i]
                    diffs_sum = subtitle_diff_sum(edge_batch, start_contour).tolist()
            else:
                assert(start_time is not None)
                assert(start_frame is not None)
                assert(start_contour is not None)
                assert(diffs_sum is not None)
                logger.info(f"[STATS] i={cur_time} diff={diffs_sum[i]}")
                # has starting frame
                if edge_sum[i] > config.empty:
                    # current frame has text
                    if diffs_sum[i] > config.diff:
                        # Current frame differs
                        logger.info(f"[EVENT]::TEXT->NEW TEXT {cur_time}")
                        yield {
                            "start": start_time,
                            "end": cur_time,
                            "frame": start_frame.cpu()
                        }
                        start_time = cur_time
                        start_frame = frame_batch[i]
                        start_contour = edge_batch[i]
                        diffs_sum = subtitle_diff_sum(edge_batch, start_contour).tolist()
                else:
                    # current frame has no text
                    logger.info(f"[EVENT]::TEXT->EMPTY {cur_time}")
                    yield {
                        "start": start_time,
                        "end": cur_time,
                        "frame": start_frame.cpu()
                    }
                    start_time = None
                    start_frame = None
                    start_contour = None

    
    if start_frame is not None:
        assert(start_contour is not None)
        yield {
            "start": start_time,
            "end": cur_time,
            "frame": start_frame
        }

def ocr_text_generator(key_frame_generator, config: OCRConfig = OCRConfig()):
    logger = logging.getLogger('KEY')
    reader = easyocr.Reader(['ch_tra', 'en'])
    for keys in batcher(key_frame_generator, config.batch):
        images = [
            key['frame'].permute([1,2,0]).numpy()
            for key in keys
        ]
        results = reader.readtext_batched(
            images,
            blocklist="~@#$%^&*()_+{}|:\"<>~`[]\\;'/"
        )
        for i, res in enumerate(results):
            logger.info(f"[OUT] {res}")
            yield {
                "start": keys[i]["start"],
                "end": keys[i]["end"],
                "ocrs": res
            }

def subtitle_from_ocr(ocr_result):
    filtered = [
        r for r in ocr_result
        if not r[1].isdigit() and r[0][2][1] - r[0][0][1] > 30
    ]
    print(filtered)
    high = max(map(lambda x:x[2], filtered))
    return zhconv.convert("\n".join([
        r[1]
        for r in filtered
        if r[2] > high / 2
        or sum(1 if '\u4e00' <= c <= '\u9fff' else 0 for c in r[1]) >= len(r[1])//3
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
                time1 = datetime.utcfromtimestamp(last_start).strftime('%H:%M:%S,%f')[:-3]
                time2 = datetime.utcfromtimestamp(last_end).strftime('%H:%M:%S,%f')[:-3]
                yield "\n".join([
                    f"{cnt}",
                    f"{time1} --> {time2}",
                    last_text
                ])
                cnt+=1
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

def debug_contour(path, config: KeyConfig = KeyConfig()):
    reader = torchvision.io.VideoReader(path, "video")

    for i, video_frame in enumerate(reader):
        frame = video_frame["data"]
        torchvision.io.write_png(frame, f"img/{i}.png")
        edge = subtitle_black_contour_single(frame, config.contour)
        if edge.int().sum() > config.empty:
            torchvision.io.write_png((edge*255).unsqueeze(0).to(torch.uint8), f"img/{i}_.png")

def debug_key(key_frame_generator, config: KeyConfig = KeyConfig()):
    for i,key in enumerate(key_frame_generator):
        frame = key["frame"]
        edge = subtitle_black_contour_single(frame, config.contour).to(torch.uint8)
        torchvision.io.write_png(frame, f"key/{i}.png")
        torchvision.io.write_png((edge*255).unsqueeze(0).to(torch.uint8), f"key/{i}_.png")