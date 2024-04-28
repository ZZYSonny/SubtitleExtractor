import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import tempfile

RSS_URL = "https://api.ani.rip/ani-download.xml"
#TEMP_DIR = tempfile.TemporaryDirectory(prefix="subs").name
TEMP_DIR = ".temp"
IN_VIDEO_PATH = osp.join(TEMP_DIR, "in.mp4")
OUT_SUBS_PATH = osp.join(TEMP_DIR, "out.srt")
OUT_VIDEO_PATH = osp.join(TEMP_DIR, "out.mkv")