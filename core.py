import logging
import torch
import torchvision
from torch.nn import functional as F
from dataclasses import dataclass, asdict
from tqdm import tqdm
import easyocr
from datetime import datetime
import zhconv
import itertools
import os

logging.basicConfig(level=logging.INFO)


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
    200, 1000, 512, 16, 10, ContourConfig(8, 8, 2, 5, 1))
KeyConfig1080p2x = KeyConfig(
    50, 250, 512, 16, 10, ContourConfig(8, 8, 2, 3, 2))
EasyOCRArgs = dict(
    blocklist="~@#$%^&*()_+{}|:\"<>~`[]\\;'/",
    batch_size=2
)


@torch.compile(mode="max-autotune")
def subtitle_black_contour(rgb: torch.Tensor, config: ContourConfig):
    channel_dim = 1
    r = rgb.select(channel_dim, 0).unsqueeze(channel_dim)
    grey_mask = rgb.eq(r).all(dim=channel_dim)
    white_mask = torch.logical_and(
        rgb.gt(255 - config.white).all(dim=channel_dim),
        grey_mask
    )
    black_mask = torch.logical_and(
        rgb.lt(config.black).all(dim=channel_dim),
        grey_mask
    )
    if config.scale == 1:
        white_cnt_scaled = white_mask.to(torch.float16)
        black_mask_scaled = black_mask
    else:
        white_cnt_scaled = F.avg_pool2d(
            white_mask.to(torch.float16),
            kernel_size=config.scale,
            padding=config.scale//2,
            stride=1,
            divisor_override=1
        )
        black_mask_scaled = F.avg_pool2d(
            black_mask.to(torch.float16),
            kernel_size=config.scale,
            padding=config.scale//2,
            stride=1,
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
            xs.shape[0] - 1
        )

    r1, r2 = bound_1d(edge.int().sum(dim=1))
    c1, c2 = bound_1d(edge.int().sum(dim=0))
    return frame[:, r1:r2, c1:c2]


def key_frame_generator(path, config: KeyConfig):
    logger = logging.getLogger('KEY')
    reader = torchvision.io.VideoReader(path, "video", num_threads=8)

    has_start = 0
    start_time = 0.0
    start_frame = torch.empty(0)
    start_edge = torch.empty(0)

    def select_key_frame(cur_time, cur_frame, cur_edge):
        nonlocal has_start, start_time, start_frame, start_edge
        assert (not has_start)
        has_start = True
        start_time = cur_time
        # Clone edge so that in the next batch, previous edge_batch
        # can be deleted.
        start_frame = subtitle_bound(
            subtitle_region(cur_frame),
            cur_edge,
            config
        ).clone()
        start_edge = cur_edge.clone()

    def release_key_frame(cur_time):
        nonlocal has_start
        assert (has_start)
        has_start = False
        return {
            "start": start_time,
            "end": cur_time,
            "frame": start_frame
        }

    def stack_frames(batch):
        return torch.stack([
            subtitle_region(frame["data"])
            for frame in batch
        ]).to(config.device)

    last_batch = []
    while True:
        logger.info("Decoding videos")
        new_batch = list(itertools.islice(reader, config.batch_edge - len(last_batch)))
        frame_batch = last_batch + new_batch

        logger.info("Computing edges")
        edge_batch = subtitle_black_contour(
            stack_frames(frame_batch), config.contour)
        empty_batch = edge_batch.int().sum(dim=[1, 2]).lt(config.empty).cpu()

        logger.info("Batch Ready")
        window_start = 0
        while True:
            # Determine current window [cur_start, cur_end]
            window_end = window_start + config.batch_window

            if window_end > len(frame_batch):
                if len(frame_batch) == config.batch_edge:
                    logger.info(
                        f"Window outside Batch. Leave {window_start}:{config.batch_edge} to next batch.")
                    last_batch = frame_batch[window_start:]
                    break
                elif window_start == len(frame_batch):
                    logger.info(f"Finished.")
                    return
                else:
                    window_end = len(frame_batch)
                    logger.info(f"Last batch and window outside batch. Do the last {window_start}:{window_end}")

            loc = f"{frame_batch[window_start]['pts']:.2f} -> {frame_batch[window_end-1]['pts']:.2f}"

            if not has_start:
                non_empty_window = torch.logical_not(
                    empty_batch[window_start:window_end])
                i = non_empty_window.nonzero_static(
                    size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info(f"{loc} Empty: No Change")
                else:
                    window_start = window_start + i
                    logger.info(
                        f"{loc} Empty -> Text at {frame_batch[window_start]['pts']}")
                    select_key_frame(
                        frame_batch[window_start]["pts"], frame_batch[window_start]["data"], edge_batch[window_start])
            else:
                # Compute diff
                edge_window = edge_batch[window_start:window_end]
                diff_window = torch.logical_xor(
                    edge_window, start_edge).int().sum(dim=[1, 2])
                changed_window = diff_window.gt(config.diff).cpu()
                empty_window = empty_batch[window_start:window_end]
                stop_window = torch.logical_or(changed_window, empty_window)

                i = stop_window.nonzero_static(size=1, fill_value=-1).item()
                if i == -1:
                    window_start = window_end
                    logger.info(f"{loc} Text: No Change")
                else:
                    window_start = window_start + i
                    cur_time = frame_batch[window_start]['pts']
                    yield release_key_frame(cur_time)

                    if empty_window[i].item():
                        logger.info(f"{loc} Text -> Empty  at {cur_time:2f}")
                    else:
                        logger.info(f"{loc} Text -> Changed at {cur_time:2f}")
                        select_key_frame(
                            cur_time, frame_batch[window_start]["data"], edge_batch[window_start])


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
    reader = torchvision.io.VideoReader(path, "video")

    for i, video_frame in enumerate(reader):
        frame = video_frame["data"]
        torchvision.io.write_png(frame, f"debug/img/{i}.png")
        edge = subtitle_black_contour_single(frame, config.contour)
        if edge.int().sum() > config.empty:
            torchvision.io.write_png(
                (edge*255).unsqueeze(0).to(torch.uint8), f"debug/img/{i}_.png")


def debug_key(key_frame_generator, config: KeyConfig):
    os.system("rm debug/key/*.png")
    for i, key in enumerate(key_frame_generator):
        frame = key["frame"]
        edge = subtitle_black_contour_single(
            frame, config.contour).to(torch.uint8)
        torchvision.io.write_png(frame, f"debug/key/{i}.png")
        torchvision.io.write_png(
            (edge*255).unsqueeze(0).to(torch.uint8), f"debug/key/{i}_.png")
