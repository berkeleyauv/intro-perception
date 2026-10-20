from pathlib import Path

import cv2

from intro_perception.classical import ClassicalGatePerceiver

path = next(Path("baseline_failures/originals").glob("1755012416_145619*.jpg"))
img = cv2.imread(str(path))
detector = ClassicalGatePerceiver()

OUT = Path("/tmp/threshold_test")
OUT.mkdir(exist_ok=True)

# the more we raise the threshold the more dark pixels we let in
#for v in [90, 95, 100, 105, 110, 130]:
    #estimate, frames = detector.analyze(img, debug=True, slider_vals={"black_v_max": v})
    #cv2.imwrite(str(OUT / f"mask_{v}.png"), frames[1])
    #cv2.imwrite(str(OUT / f"annotated_{v}.png"), frames[0])
    
    #print(v, estimate.visible, round(estimate.confidence, 2),
          #round(estimate.box_x, 2), round(estimate.box_y, 2),
          #round(estimate.box_width, 2), round(estimate.box_height, 2))
# starting to let in unnecessary pixels with 110, 130
# BUT just ran it so that it annotates on OG photos and boundaries from 105-130 captured the gate exactly



detector = ClassicalGatePerceiver()
for photo in sorted(Path("baseline_failures/originals").glob("*.jpg")):
    img = cv2.imread(str(photo))
    print(photo.name.split("_jpg")[0])
    for v in [50, 105, 115, 130]:
        e = detector.analyze(img, debug=False, slider_vals={"black_v_max": v})
        print("  ", v, "->", "gate" if e.visible else "no gate", round(e.confidence, 2),
              "box:", round(e.box_x, 2), round(e.box_y, 2), round(e.box_width, 2), round(e.box_height, 2))