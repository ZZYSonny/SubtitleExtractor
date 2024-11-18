import torch
import queue
import threading

from . import stages

config = stages.FullConfig(
    exe=stages.ExecConfig(batch=1, device="cuda"),
    key=stages.KeyConfig(empty_ratio=0.003, diff_ratio=0.4, diff_cd=0.8),
    box=stages.CropConfig(
        top=856, down=0, left=192, right=192, width=1920, height=1080
    ),
    filter=stages.FilterConfig(
        range_y_white=19,
        range_uv_grey=1,
    ),
    ocr=dict(
        # https://www.jaided.ai/easyocr/documentation/
        blocklist=" `~@#$%^&*_+={}[]|\\:;<>/",
        batch_size=16,
        contrast_ths=0,
        # https://github.com/clovaai/CRAFT-pytorch/issues/51
        # text_threshold=0.3,
        # low_text=0.2
    ),
    sub=stages.SubsConfig(min_conf=0.2, fix_delta_sec=-0.01, merge_max_sec=0.1),
)


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
            if x is None:
                break
            else:
                yield x

    if isinstance(xs, list):
        return xs
    else:
        return async_generator()


def convert_subtitle(in_video_path: str, out_sub_path: str):
    keys = async_iterable(stages.key_frame_generator(in_video_path, config))
    ocrs = async_iterable(stages.ocr_text_generator(keys, config))
    srts = stages.srt_generator(out_sub_path, ocrs, config)
    torch.cuda.empty_cache()
