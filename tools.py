import os
os.system("rm -rf /home/zzysonny/Documents/Code/Projects/SubtitleExtractor/debug/img/*")
os.system("rm -rf /home/zzysonny/Documents/Code/Projects/SubtitleExtractor/debug/error/*")
import warnings
warnings.filterwarnings('ignore')
#import os
#os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"]="1"
#os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"]="1"
#os.environ["TORCHINDUCTOR_LAYOUT_OPTIMIZATION"]="1"
#os.environ["TORCHINDUCTOR_DEBUG_FUSION"]="1"
#os.environ["TORCHINDUCTOR_PERMUTE_FUSION"]="1"
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

RTX2060Config = SubsConfig(
    exe = ExecConfig(
        batch = 3,
        device = "cuda"
    ),
    key = KeyConfig(
        empty=0.003, 
        diff_tol=0.4,
    ),
    box = CropConfig(
        top=888,
        down=0,
        left=192,
        right=192,
        width=1920,
        height=1080
    ),
    contour=ContourConfig(
        y_black_tol=16,
        y_white_tol=32, 
        uv_tol=2,
        white_x_scale=4,
        white_y_scale=4,
        white_min=1,
        black_x_scale=16,
        black_y_scale=64,
        black_min=1,
        abs_min_x=2,
        abs_min_y=2,
    ),
    ocr = dict(
        # https://www.jaided.ai/easyocr/documentation/
        blocklist=" `~@#$%^&*_+={}[]|\\:;<>/",
        batch_size=16,
        contrast_ths=0,
        # https://github.com/clovaai/CRAFT-pytorch/issues/51
        #text_threshold=0.3,
        #low_text=0.2
    )
)
config = RTX2060Config

def get_anime_link_from_ani_xml(name: str):
    # 获取RSS
    proxy = urllib.request.ProxyHandler(urllib.request.getproxies())
    opener = urllib.request.build_opener(proxy)
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    print("下载RSS")
    urllib.request.urlretrieve(RSS_URL, filename=RSS_PATH)

    for item in reversed(ET.parse(RSS_PATH).getroot().findall('./channel/item')):
        title = item.find("title").text
        link = item.find("link").text
        size_elem = [child for child in item if child.tag.endswith("size")]
        size = None
        if len(size_elem)==1:
            size = int(float(size_elem[0].text.split(" ")[0])*1024*1024)

        if name in title:
            print(f"发现视频 {title}")
            return link, size

    raise Exception("未发现视频")

def get_anime_link_from_ani_folder(folder: str):
    pass

def download_anime_from_link(link: str, size: int | None = None):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, total=size,
                             miniters=1, desc="下载视频") as t:
        urllib.request.urlretrieve(
            link, filename=IN_VIDEO_PATH, reporthook=t.update_to)



def convert_subtitle():
    keys = async_iterable(key_frame_generator(IN_VIDEO_PATH, config))
    ocrs = async_iterable(ocr_text_generator(keys, config))
    srts = list(srt_entry_generator(ocrs))
    with open(OUT_SUBTITLE_PATH, "w") as f:
        print("\n\n".join(srts), file=f)
    torch.cuda.empty_cache()

    
def merge_and_serve():
    os.system(" ".join([
        f"ffmpeg -y",
        f"-i {IN_VIDEO_PATH}",
        f"-i {OUT_SUBTITLE_PATH}",
        f"-c copy",
        f"-metadata:s:s:0 language=zh-CN",
        f"-disposition:s:0 default",
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

