import os
import os.path as osp
import requests
import urllib.parse
import json

HEADER = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "sec-ch-ua": "\"Chromium\";v=\"123\", \"Not:A-Brand\";v=\"8\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Linux\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "cookie": "cf_clearance=25s8ElJCHaqWIVfwl0Kh0gpgIM6EQJQfQVpkMfoN8N0-1714412428-1.0.1.1-BYiPo8rQIoo2wLkjPnLrKHdO._Zbg9VyAI3xJ7Kxi3CR5ToNoFjCVZ8Hcv6bU.6TSOS2eE4ru8Nr5nzv67lH.g",
}

def get_link_from_search(name: str):
    encoded_name = urllib.parse.quote(name)
    response = requests.get(
        f"https://search.acgn.es/api/?cid=2&page=0&limit=24&word={encoded_name}&sort=time&file_suffix=",
        headers=HEADER
    )
    animes = sorted(json.loads(response.text)["data"], key=lambda r:r['date'], reverse = True)
    print(animes[0]["text"])
    return animes[0]["link"]

def download(path: str, link: str):
    fdir = osp.dirname(osp.abspath(path))
    fname = osp.basename(path)
    os.system(" ".join([
        f"/usr/bin/tdl dl",
        f"-u {link}",
        f"-d {fdir}",
        f"--template '{fname}'"
    ]))
