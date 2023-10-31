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

## 高级
TODO