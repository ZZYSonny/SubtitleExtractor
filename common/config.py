import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import tempfile
import atexit
import shutil

TEMP_DIR = ""
IN_VIDEO_PATH = ""
IN_SUBS_PATH = ""
OUT_VIDEO_PATH = ""
OUT_SUBS_PATH = ""

def select_temp_dir(dir: str):
    global TEMP_DIR, IN_VIDEO_PATH, IN_SUBS_PATH, OUT_VIDEO_PATH, OUT_SUBS_PATH
    TEMP_DIR = dir
    IN_VIDEO_PATH = osp.join(TEMP_DIR, "in.mp4")
    IN_SUBS_PATH = osp.join(TEMP_DIR, "in.srt")
    OUT_VIDEO_PATH = osp.join(TEMP_DIR, "out.mkv")
    OUT_SUBS_PATH = osp.join(TEMP_DIR, "out.srt")

def create_temp_dir():
    dir = tempfile.mkdtemp(prefix="subs")
    atexit.register(lambda: shutil.rmtree(dir))
    select_temp_dir(dir)

create_temp_dir()