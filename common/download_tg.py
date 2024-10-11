import os
import os.path as osp
import requests
import urllib.parse
import json

HEADER = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "priority": "u=1, i",
    "sec-ch-ua": '"Chromium";v="125", "Not.A/Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "cookie": "cf_clearance=25s8ElJCHaqWIVfwl0Kh0gpgIM6EQJQfQVpkMfoN8N0-1714412428-1.0.1.1-BYiPo8rQIoo2wLkjPnLrKHdO._Zbg9VyAI3xJ7Kxi3CR5ToNoFjCVZ8Hcv6bU.6TSOS2eE4ru8Nr5nzv67lH.g; cf_clearance=n9q5atoHMr4f10Jae5eVrDBzD3qmGCNjzJJ98N7SVqc-1717430780-1.0.1.1-uhurB9IhRTcn1dYdoDAKuzU73Bq4bWwtMzG0G.Wo05KWHqw8V8LQCjQ3bpBWcLwq78TjmXsVBC_czII7vcvlog",
    "Referer": "https://search.acgn.es/?cid=2&word=%E5%BE%9E+Lv2+%E9%96%8B%E5%A7%8B%E9%96%8B%E5%A4%96%E6%8E%9B%E7%9A%84%E5%89%8D%E5%8B%87%E8%80%85%E5%80%99%E8%A3%9C%E9%81%8E%E8%91%97%E6%82%A0%E5%93%89%E7%95%B0%E4%B8%96%E7%95%8C%E7%94%9F%E6%B4%BB&sort=time&file_suffix=",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


def get_link_from_search(name: str):
    encoded_name = urllib.parse.quote(name)
    response = requests.get(
        f"https://search.acgn.es/api/?cid=2&page=0&limit=24&word={encoded_name}&sort=time&file_suffix=",
        headers=HEADER,
    )
    animes = sorted(
        json.loads(response.text)["data"], key=lambda r: r["date"], reverse=True
    )
    print(animes[0]["text"])
    return animes[0]["link"]


def download(path: str, link: str):
    fdir = osp.dirname(osp.abspath(path))
    fname = osp.basename(path)
    os.system(
        " ".join(
            [
                f"/usr/bin/tdl dl",
                f"-u {link}",
                f"-d {fdir}",
                f"--template '{fname}'",
                # f"--continue"
            ]
        )
    )
