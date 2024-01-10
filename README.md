## 介绍
识别巴哈番剧的内嵌繁体字幕，并生成简体外挂字幕。经测试，在R7-4800HS，RTX2060的笔记本上，处理一集仅需2分钟。

## 使用
- 安装conda环境
```py
conda env create -f environment.yml
```

- 翻到`main.py`的最底下，填写番剧名称，例如：
```py
name = "星靈"
```

- 运行。
```bash
conda activate subs
python3 main.py
```

## FAQ
### 能在CPU上跑吗
~~最早是同时支持CPU和GPU的，我还想着怎么优化让CPU跑快点。结果写着写着发现视频解码是个瓶颈，用上硬件解码能快很多，就回不去了。~~

### 我的显存太多/太少了，能调吗
`main.py`自带的参数需要6GB显存。如需适配其他显卡，可以调整batch size。项目中有两个batch size需要调整。一个是识别关键帧的batch_edge，默认一次用显卡解码512帧。另一个是OCR模型的batch_size，默认是16。
```py
KeyExtractorConfig1080p1x = KeyConfig(
    empty=200, 
    diff=1000, 
    batch_edge=512, 
    batch_window=16, 
    margin=10, 
    contour=ContourConfig(
        white=32, 
        black=32, 
        near=2, 
        kernel=5, 
        scale=1
    )
)
KeyExtractorConfig = KeyExtractorConfig1080p1x
EasyOCRArgs = dict(
    blocklist="~@#$%^&*_-+={}[]|\\:;<>/",
    batch_size=16
)
```

### 字幕出现问题，怎么调试
如果字幕在该更新的时候没有更新，那么很有可能是关键帧识别的bug。
- 首先用ffmpeg把出问题的一段切出来。
```bash
ffmpeg -i in0.mp4 -ss 00:22:20 -t 00:02:00 -c:v copy -c:a copy in.mp4
```
- 开启调试日志，再次运行`main.py`
```bash
export LOGLEVEL=DEBUG
python3 main.py
```
- 为了方便找到帧以及对应的Contour,可以用这个函数。
```py
debug_contour(IN_VIDEO_PATH, KeyExtractorConfig)
```

如果字幕识别出错，那么就去调EasyOCR的参数吧。

###
```bash
export ALL_PROXY=socks5://127.0.0.1:10808
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY=http://127.0.0.1:10809
```