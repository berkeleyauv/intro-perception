from pathlib import Path

import cv2
import numpy as np

path = next(Path("baseline_failures/originals").glob("1755012416_145619*.jpg"))
img = cv2.imread(str(path))
v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]   # the brightness channel

print("darkest pixel:", v.min())
print("1st / 5th / 50th percentile:", np.percentile(v, [1, 5, 50]))