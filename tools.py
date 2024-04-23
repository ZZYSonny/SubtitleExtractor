import warnings
warnings.filterwarnings('ignore')
import functools
import os.path as osp
import urllib.request
import xml.etree.ElementTree as ET
from tqdm import tqdm
from core import *
import http.server
import socket
import socketserver
import requests
import json
import tempfile

HEADER = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": "\"Chromium\";v=\"123\", \"Not:A-Brand\";v=\"8\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Linux\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1"
}
RSS_URL = "https://api.ani.rip/ani-download.xml"
#TEMP_DIR = tempfile.TemporaryDirectory(prefix="subs").name
TEMP_DIR = "./temp"
IN_VIDEO_PATH = osp.join(TEMP_DIR, "in.mp4")
OUT_SUBTITLE_PATH = osp.join(TEMP_DIR, "in.mp4")
OUT_VIDEO_PATH = osp.join(TEMP_DIR, "in.mp4")
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
    print("下载RSS")
    response = requests.get(f'https://api.ani.rip/ani-download.xml', headers=HEADER)

    for item in reversed(ET.ElementTree(ET.fromstring(response.text)).findall("./channel/item")):
        title = item.find("title").text
        link = item.find("link").text
        size = float(item.find("{https://open.ani-download.workers.dev}size").text.split(" ")[0])
        if name in title:
            print(f"发现视频 {title}")
            return link, size

    raise Exception("未发现视频")

def get_anime_link_from_ani_folder(folder: str, name:str):
    data = '{"password":"null"}'
    print("下载File List")
    response = requests.post(f'https://aniopen.an-i.workers.dev/{folder}/', headers=HEADER, data=data)

    for info in json.loads(response.text)['files']:
        if name in info["name"]:
            print(f"发现视频 {info['name']}")
            encoded = urllib.parse.quote(info['name'])
            url = f"https://aniopen.an-i.workers.dev/{folder}/{encoded}"
            size = int(float(info["size"])/1024/1024)
            return url, size
    raise Exception("未发现视频")

def download_anime_from_link(link: str, size: int | None = None):
    response = requests.get(link, headers=HEADER, stream=True)
    pbar = tqdm(desc="下载视频", total=size*1024*1024, unit='B', unit_scale=True)

    with open(IN_VIDEO_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()

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

