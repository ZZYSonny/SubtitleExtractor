import requests
import json
import urllib.parse
import xml.etree.ElementTree as ET

from tqdm import tqdm

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

def download_anime_from_link(path: str, link: str, size: int | None = None):
    response = requests.get(link, headers=HEADER, stream=True)
    pbar = tqdm(desc="下载视频", total=size*1024*1024, unit='B', unit_scale=True)

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()
