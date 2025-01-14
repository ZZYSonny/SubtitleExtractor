from common import config, download_ani, ffmpeg, serve
from ocr_baha import pipeline


if False:
    config.select_temp_dir(".temp")
elif False:
    config.select_temp_dir("/tmp/subs_8emu3z6")
elif False:
    link = "https://resources.ani.rip/2024-10/%5BANi%5D%20%E8%86%BD%E5%A4%A7%E9%BB%A8%20-%2001%20%5B1080P%5D%5BBaha%5D%5BWEB-DL%5D%5BAAC%20AVC%5D%5BCHT%5D.mp4?d=true"
    download_ani.download(config.IN_VIDEO_PATH, link, 1000)
elif True:
    try:
        # https://api.ani.rip/ani-download.xml
        name = "推理病歷表"
        link, size = download_ani.get_link_from_xml(name)
    except Exception as e:
        link, size = download_ani.get_link_from_folder("2024-10", name)
    download_ani.download(config.IN_VIDEO_PATH, link, size)

pipeline.convert_subtitle(config.IN_VIDEO_PATH, config.OUT_SUBS_PATH)
ffmpeg.replace_subs(config.IN_VIDEO_PATH, config.OUT_SUBS_PATH, config.OUT_VIDEO_PATH)
serve.open_in_explorer(config.TEMP_DIR)
serve.serve(config.OUT_VIDEO_PATH)
