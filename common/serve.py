import os.path as osp
import socket
import functools
import socketserver
import http.server

def serve(out_video_path: str):
    ip = socket.gethostbyname(socket.gethostname())
    folder_abs_path = osp.abspath(osp.join(out_video_path, ".."))
    out_base_name = osp.basename(out_video_path)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder_abs_path)
    with socketserver.TCPServer(("", 8000), handler) as httpd:
        print(f"转换完成,视频可通过 http://{ip}:8000/{out_base_name} 下载")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("服务器关闭")