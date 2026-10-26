"""
Prepare the portrait for ASCII conversion.

1. Remove the background (rembg, human segmentation model).
2. Crop to head + crossed arms.
3. Grayscale + CLAHE local contrast so face, beard and shirt read well.
4. Save gray + alpha mask -> background becomes blank space in the ASCII.

usage: python scripts/prep_photo.py photo.jpg source-prepped.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

# crop box as fractions of the original (left, top, right, bottom)
CROP = (0.06, 0.10, 0.90, 0.78)

src = Image.open(INP).convert("RGB")

try:
    from rembg import new_session, remove
    cut = remove(src, session=new_session("u2net_human_seg"))
except ImportError:
    sys.exit("pip install rembg onnxruntime  (needed to remove the background)")

w, h = cut.size
cut = cut.crop((int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h)))

rgba = np.array(cut)
gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
gray = clahe.apply(gray)

alpha = rgba[:, :, 3].astype(np.float32) / 255.0
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
# keep the subject mask in the alpha channel: make_ascii_svg.py uses it to
# blank the background, so the ink can follow brightness (positive image).
la = np.dstack([gray, (alpha * 255).clip(0, 255).astype(np.uint8)])
Image.fromarray(la, "LA").save(OUT)
print("wrote", OUT, Image.open(OUT).size)
