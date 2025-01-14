from llm_translate import pipeline_context
from common import config, ffmpeg, serve, download_tg

config.select_temp_dir(".temp")

name = "終末的火車前往何方"
pipeline_context.config.prompt_map["lang_source"] = "英文"
pipeline_context.config.prompt_map["history"] = (
    "郊外的某个小镇。这里不是一个随处可见的普通乡村……。这里的居民产生了巨大的异变。即使在这样的情况下，主角也有着坚定的想法。想再一次见到行踪不明的朋友们！她们在被废弃且停止行驶的电车里，在不知道能不能活着回来的情况下前往外面的世界。开始运行的末日列车的终点，到底有什么？"
)
pipeline_context.config.prompt_map["no_translate"] = (
    "Yoka / Yoka-chan / Nadeko-chan / Shizuru-chan / Pontaro / Pochi / Pochi-san / Ikebukuro"
)
# link = download_tg.get_link_from_search(name)
# download_tg.download(config.IN_VIDEO_PATH, link)

ffmpeg.extract_subs(config.IN_VIDEO_PATH, config.IN_SUBS_PATH)
pipeline_context.translate_srt_vllm(
    config.IN_SUBS_PATH, config.OUT_SUBS_PATH, pipeline_context.config
)
ffmpeg.prepend_subs(config.IN_VIDEO_PATH, config.OUT_SUBS_PATH, config.OUT_VIDEO_PATH)
serve.open_in_explorer(config.TEMP_DIR)
serve.serve(config.OUT_VIDEO_PATH)
