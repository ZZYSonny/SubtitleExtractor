import os
import os.path as osp


def extract_subs(in_video_path: str, in_subs_path: str):
    os.system(
        " ".join(
            [
                f"ffmpeg -y",
                f"-loglevel error",
                f"-i {in_video_path}",
                f"-map 0:s:m:language:eng",
                f"-c:s text",
                f"{in_subs_path}",
            ]
        )
    )


def replace_subs(in_video_path: str, out_subs_path: str, out_video_path: str):
    print("Merging")
    os.system(
        " ".join(
            [
                f"ffmpeg -y",
                f"-loglevel error",
                f"-i {in_video_path}",
                f"-i {out_subs_path}",
                f"-c copy",
                f"-metadata:s:s:0 language=zh-CN",
                f"-disposition:s:0 default",
                f"{out_video_path}",
            ]
        )
    )
    print(f"转换完成,视频保存在 {osp.abspath(osp.join(out_video_path, '..'))}")


def prepend_subs(in_video_path: str, out_subs_path: str, out_video_path: str):
    print("Merging")
    os.system(
        " ".join(
            [
                f"ffmpeg -y",
                f"-loglevel error",
                f"-i {in_video_path}",
                f"-i {out_subs_path}",
                f"-map 1",
                f"-c copy",
                f"-map 0",
                f"-c copy",
                f"-metadata:s:s:0 language=zh-CN",
                f"-disposition:s:0 default",
                f"{out_video_path}",
            ]
        )
    )
    print(f"转换完成,视频保存在 {osp.abspath(osp.join(out_video_path, '..'))}")
