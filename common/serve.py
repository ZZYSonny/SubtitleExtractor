import os
import os.path as osp
import socket
import functools
import socketserver
import http.server

def merge(in_video_path: str, out_subs_path: str, out_video_path: str):
    os.system(" ".join([
        f"ffmpeg -y",
        f"-i {in_video_path}",
        f"-i {out_subs_path}",
        f"-c copy",
        f"-metadata:s:s:0 language=zh-CN",
        f"-disposition:s:0 default",
        f"{out_video_path}"
    ]))
    print(f"转换完成,视频保存在 {os.path.abspath(out_video_path)}")

def serve(out_video_path: str):
    ip = socket.gethostbyname(socket.gethostname())
    folder_abs_path = osp.abspath(osp.join(out_video_path, ".."))
    out_base_name = osp.basename(out_video_path)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder_abs_path)
    with socketserver.TCPServer(("", 8000), handler) as httpd:
        print(f"转换完成,视频可通过 http://{ip}:8000/{out_base_name} 下载")
        httpd.serve_forever()