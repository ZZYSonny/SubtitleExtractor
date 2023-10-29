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
    scale: int


@dataclass
class KeyConfig:
    empty: int
    diff: int
    batch: int
    margin: int
    contour: ContourConfig
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


KeyConfig1080p1x = KeyConfig(200, 1000, 512, 10, ContourConfig(8, 8, 2, 5, 1))
KeyConfig1080p2x = KeyConfig(50, 250, 512, 10, ContourConfig(8, 8, 2, 3, 2))
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
    def bound_1d(xs,):
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


def subtitle_diff_sum(contours, reference):
    return torch.logical_xor(contours, reference).int().sum(dim=[1, 2])


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


def key_frame_generator(path, config: KeyConfig):
    logger = logging.getLogger('KEY')
    reader = torchvision.io.VideoReader(path, "video", num_threads=8)

    start_time = 0.0
    start_frame = None
    start_contour = None

    for video_frames in batcher(reader, config.batch):
        edge_batch = subtitle_black_contour(torch.stack([
            subtitle_region(frame["data"])
            for frame in video_frames
        ]).to(config.device), config.contour)
        edge_sum = edge_batch.int().sum(dim=[1, 2]).tolist()

        if start_contour is not None:
            diffs_sum = subtitle_diff_sum(edge_batch, start_contour).tolist()
        else:
            diffs_sum = None

        for i in range(len(video_frames)):
            cur_time = video_frames[i]["pts"]
            if start_contour is None:
                logger.info(f"[STATS] i={cur_time} pix={edge_sum[i]}")
                # no starting frame
                if edge_sum[i] > config.empty:
                    # current frame has text
                    logger.info(f"[EVENT]::EMPTY->TEXT {cur_time}")
                    start_time = cur_time
                    start_frame = subtitle_bound(
                        subtitle_region(video_frames[i]["data"]), edge_batch[i], config).cpu()
                    start_contour = edge_batch[i].clone()
                    diffs_sum = subtitle_diff_sum(
                        edge_batch, start_contour).tolist()
            else:
                assert (start_time is not None)
                assert (start_frame is not None)
                assert (start_contour is not None)
                assert (diffs_sum is not None)
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
                            "frame": start_frame
                        }
                        start_time = cur_time
                        start_frame = subtitle_bound(
                            subtitle_region(video_frames[i]["data"]), edge_batch[i], config).cpu()
                        start_contour = edge_batch[i].clone()
                        diffs_sum = subtitle_diff_sum(
                            edge_batch, start_contour).tolist()
                else:
                    # current frame has no text
                    logger.info(f"[EVENT]::TEXT->EMPTY {cur_time}")
                    yield {
                        "start": start_time,
                        "end": cur_time,
                        "frame": start_frame
                    }
                    start_time = None
                    start_frame = None
                    start_contour = None

    if start_frame is not None:
        assert (start_contour is not None)
        yield {
            "start": start_time,
            "end": cur_time,
            "frame": start_frame
        }


def ocr_text_generator(key_frame_generator, easyocr_args: dict):
    logger = logging.getLogger('KEY')
    reader = easyocr.Reader(['ch_tra', 'en'])
    for key in key_frame_generator:
        image = key['frame'].permute([1, 2, 0]).numpy()
        res = reader.readtext(image,**easyocr_args)
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
    reader = torchvision.io.VideoReader(path, "video")

    for i, video_frame in enumerate(reader):
        frame = video_frame["data"]
        torchvision.io.write_png(frame, f"img/{i}.png")
        edge = subtitle_black_contour_single(frame, config.contour)
        if edge.int().sum() > config.empty:
            torchvision.io.write_png(
                (edge*255).unsqueeze(0).to(torch.uint8), f"img/{i}_.png")


def debug_key(key_frame_generator, config: KeyConfig):
    for i, key in enumerate(key_frame_generator):
        frame = key["frame"]
        edge = subtitle_black_contour_single(
            frame, config.contour).to(torch.uint8)
        torchvision.io.write_png(frame, f"key/{i}.png")
        torchvision.io.write_png(
            (edge*255).unsqueeze(0).to(torch.uint8), f"key/{i}_.png")
