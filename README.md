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