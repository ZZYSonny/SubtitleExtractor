from core import *
import pickle
import os

#video_chs_path = "video/chs.mkv"
video_cht_path = "video/cht_dbg.mp4"
video_out_path = "video/out.mkv"
srt_out_path = "video/out.srt"

key_config = KeyConfig1080p2x
ocr_config = EasyOCRArgs

#debug_contour(video_cht_path, key_config)

keys = list(key_frame_generator(video_cht_path, key_config))
debug_key(keys, key_config)
#pickle.dump(keys, open("debug/pkl/keys.pkl", "wb"))
#keys = pickle.load(open("debug/pkl/keys.pkl", "rb"))


ocrs = list(ocr_text_generator(keys, ocr_config))
##pickle.dump(ocrs, open("debug/pkl/ocrs.pkl", "wb"))
##ocrs = pickle.load(open("debug/pkl/ocrs.pkl", "rb"))
#
srts = list(srt_entry_generator(ocrs))
with open(srt_out_path, "w") as f:
    print("\n\n".join(srts), file=f)

#os.system(" ".join([
#    f"ffmpeg -y",
#    f"-i {video_cht_path}",
#    f"-sub_charenc 'UTF-8'",
#    f"-f srt -i {srt_out_path}",
#    f"-map 0:0 -map 0:1 -map 1:0 -c:v copy -c:a copy",
#    f"-c:s srt -metadata:s:s:0 language=zh-CN",
#    f"{video_out_path}"
#]))