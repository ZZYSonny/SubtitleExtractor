import os
import os.path as osp

def extract_subs(in_video_path: str, in_subs_path:str):
    os.system(" ".join([
        f"ffmpeg -y",
        f"-i {in_video_path}",
        f"-c:s text",
        f"{in_subs_path}"
    ]))


def merge_subs(in_video_path: str, out_subs_path: str, out_video_path: str):
    os.system(" ".join([
        f"ffmpeg -y",
        f"-i {in_video_path}",
        f"-i {out_subs_path}",
        f"-c copy",
        f"-metadata:s:s:0 language=zh-CN",
        f"-disposition:s:0 default",
        f"{out_video_path}"
    ]))
    print(f"转换完成,视频保存在 {osp.abspath(out_video_path)}")
