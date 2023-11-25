import functools
import os.path as osp
import urllib.request
import xml.etree.ElementTree as ET
from tqdm import tqdm
from core import *
import http.server
import socket
import socketserver

RSS_URL = "https://api.ani.rip/ani-download.xml"
RSS_PATH = "temp/rss.xml"
IN_VIDEO_PATH = "temp/in.mp4"
OUT_SUBTITLE_PATH = "temp/out.srt"
OUT_VIDEO_PATH = "temp/out.mkv"
SERVE_HTTP = True

KeyExtractorConfig1080p1x = KeyConfig(
    empty=0.002, 
    diff_tol=0.5,
    batch_edge=512, 
    batch_window=16, 
    margin=2, 
    contour=ContourConfig(
        y_tol=32, 
        uv_tol=2, 
        white_scale=4,
        white_min=2,
        black_scale=16,
        black_min=1,
    )
)
KeyExtractorConfig = KeyExtractorConfig1080p1x
EasyOCRArgs = dict(
    # https://www.jaided.ai/easyocr/documentation/
    blocklist="`~@#$%^&*_+={}[]|\\:;<>/",
    batch_size=16,
    #contrast_ths=0.5,
    #adjust_contrast=0.7,
    # https://github.com/clovaai/CRAFT-pytorch/issues/51
    #text_threshold=0.3,
    #low_text=0.2
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        self.update(b * bsize - self.n)


def download_anime_by_name(name: str):
    # 获取RSS
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc="下载RSS") as t:
        urllib.request.urlretrieve(
            RSS_URL, filename=RSS_PATH, reporthook=t.update_to)

    for item in ET.parse(RSS_PATH).getroot().findall('./channel/item'):
        title = item.find("title").text
        link = item.find("link").text
        size_elem = [child for child in item if child.tag.endswith("size")]
        size = None
        if len(size_elem)==1:
            size = int(float(size_elem[0].text.split(" ")[0])*1024*1024)

        if name in title:
            print(f"发现视频 {title}")
            with DownloadProgressBar(unit='B', unit_scale=True, total=size,
                                     miniters=1, desc="下载视频") as t:
                urllib.request.urlretrieve(
                    link, filename=IN_VIDEO_PATH, reporthook=t.update_to)
            return

    raise Exception("未发现视频")


def convert_subtitle():
    keys = list(key_frame_generator(IN_VIDEO_PATH, KeyExtractorConfig))
    #torch.cuda.empty_cache()
    ocrs = list(ocr_text_generator(keys, EasyOCRArgs))
    torch.cuda.empty_cache()
    srts = list(srt_entry_generator(ocrs))
    with open(OUT_SUBTITLE_PATH, "w") as f:
        print("\n\n".join(srts), file=f)

    
def merge_and_serve():
    os.system(" ".join([
        f"ffmpeg -y",
        f"-i {IN_VIDEO_PATH}",
        f"-sub_charenc 'UTF-8'",
        f"-f srt -i {OUT_SUBTITLE_PATH}",
        f"-map 0:0 -map 0:1 -map 1:0 -c:v copy -c:a copy",
        f"-c:s srt -metadata:s:s:0 language=zh-CN",
        f"{OUT_VIDEO_PATH}"
    ]))
    print(f"转换完成,视频保存在 {os.path.abspath(OUT_VIDEO_PATH)}")

    if SERVE_HTTP:
        ip = socket.gethostbyname(socket.gethostname())
        folder_abs_path = osp.abspath(osp.join(OUT_VIDEO_PATH, ".."))
        out_base_name = osp.basename(OUT_VIDEO_PATH)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder_abs_path)
        with socketserver.TCPServer(("", 8000), handler) as httpd:
            print(f"转换完成,视频可通过 http://{ip}:8000/{out_base_name} 下载")
            httpd.serve_forever()
#星
#16bit
#娘
#芙
#家
name = "家"
download_anime_by_name(name)
convert_subtitle()
merge_and_serve()

#debug_contour(IN_VIDEO_PATH, KeyExtractorConfig)