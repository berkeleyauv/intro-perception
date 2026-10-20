from pathlib import Path

import cv2

from intro_perception.classical import ClassicalGatePerceiver

OUT = Path("/tmp/debug_out")
OUT.mkdir(exist_ok=True)

detector = ClassicalGatePerceiver()
# slider_vals={"black_v_max": 115}

for photo in sorted(Path("baseline_failures/originals").glob("*.jpg")):
    img = cv2.imread(str(photo))
    estimate, frames = detector.analyze(img, debug=True)   # debug=True is the switch
    name = photo.name.split("_jpg")[0]
    cv2.imwrite(str(OUT / f"{name}_annotated.png"), frames[0])
    cv2.imwrite(str(OUT / f"{name}_mask.png"), frames[1])
    print(name, "->", estimate)