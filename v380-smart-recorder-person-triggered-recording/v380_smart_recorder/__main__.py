import os


# Keep OpenCV/Torch/Ultralytics from over-initializing native thread runtimes on
# some Windows systems. These must be set before importing the GUI app.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ultralytics import YOLO

PRELOADED_MODEL = YOLO("yolov8n.pt")

from .app import main


if __name__ == "__main__":
    main(PRELOADED_MODEL)
