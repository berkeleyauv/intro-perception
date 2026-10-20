# Failure Analysis


## Three failure modes: 
- Classical on cut-off gates
- Classical on washed-out photos
- YOLO on zoomed-in photos (No other issues w/ YOLO; I'm operating under the assumption that it doesn't have false-positives because the original images were mislabeled after manually looking through the annotations)

## Accuracy vs speed: 
- mAP50 0.906 vs 0.089, FPS 28 vs 796 (not taking into consideration hardware caveat)

## Training-data coverage and YOLO: 
- It learned from 1,467 images (274 with no gate). 
- If zoomed-in photos are rare in training, that would explain why they fail. Check that with the manifest before you claim it. Also mention the leakage risk.

## Classical Failures
- When both posts visible, 
- Reflection of posts still causing issues
- Dull colors, grainy photos, etc

## What to improve next
- YOLO seems to only be struggling with zoomed-in training photos, implementing a solution on those photos will make the accuracy near perfect
- Training YOLO on a data without leakage

## Side-by-side comparison

| Classical | YOLO |
|---|---|
| ![](side_by_side/photo1_classical.jpg) | ![](side_by_side/photo1_yolo.jpg) |
*(Classical fails, Yolo detects the gate)*
| ![](side_by_side/photo2_classical.jpg) | ![](side_by_side/photo2_yolo.jpg) |
*(Both Yolo and Classical fail here)*

### YOLO failures

![YOLO failure](failures/yolo/1755012486_780391_jpg.rf.5b0b1996ec9c06c589067b18242c98b5.jpg)
*The photo is so zoomed in that the gate fills the frame and YOLO misses it.*

![YOLO failure](failures/yolo/1755028048_746545_jpg.rf.a8ce802927be6af41f0fb1cc3c4536d8.jpg)
*Gate is too far away for YOLO to detect it*

![YOLO failure](failures/yolo/1755028373_084966_jpg.rf.9273d9f613a95f20c215e15db8db167d.jpg)
*Right post in frame but YOLO misses it, possibly because it's too zoomed in*

## Classical failures

![Classical failure](failures/classical/1755012416_145619_jpg.rf.72a65fdf98f41a1d5d8b677f0bc19671.jpg)
*Missed, new classical method is very buggy*

![Classical failure](failures/classical/1755027387_91979_jpg.rf.743eba6e5fe7c8b0ba7292678b2b49c6.jpg)
*Miss, detecting random black edges as too small for there to be a gate..*

![Classical failure](failures/classical/1755027985_488229_jpg.rf.910e2d80c95123ab92ea4f859d02803b.jpg)
*Miss, detected reflection blobs instead of the post..*

