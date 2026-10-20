from pathlib import Path

import cv2
import numpy as np

from intro_perception.classical import ClassicalGatePerceiver

detector = ClassicalGatePerceiver()
folder = Path("baseline_failures/originals")

# Part 1: photo 3 at black_v_max = 115, saving the annotated image and the mask
photo3 = next(folder.glob("1755027964_094698*.jpg"))
img = cv2.imread(str(photo3))
estimate, frames = detector.analyze(img, debug=True, slider_vals={"black_v_max": 115})
OUT = Path("/tmp/photo3_115")
OUT.mkdir(exist_ok=True)
cv2.imwrite(str(OUT / "annotated.png"), frames[0])
cv2.imwrite(str(OUT / "mask.png"), frames[1])
print("photo 3 at 115:", estimate)

# Part 2: brightness percentiles for each failure photo
print()
print("brightness (1st, 5th, 25th, 50th percentile):")
for photo in sorted(folder.glob("*.jpg")):
    v = cv2.cvtColor(cv2.imread(str(photo)), cv2.COLOR_BGR2HSV)[:, :, 2]
    print(photo.name.split("_jpg")[0], np.percentile(v, [1, 5, 25, 50]).round())