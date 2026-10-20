# Classical Method vs YOLO Comparison

## Dataset

Completed training on 1,467 images (274 with no gate), val 183 (36 no gate), test 183 (71 truncated, 112 whole; 140 gates)

| Split | Whole gate | Cut-off gate | No gate | Total images |
|---|---|---|---|---|
| Train | 574 | 619 | 274 | 1,467 |
| Val | 68 | 79 | 36 | 183 |
| Test | 69 | 71 | 43 | 183 |

## Methods

Classical:
- Tweaked the color checks so that more dull shades of red are accepted
- Tweaked the color checks so that more black objects are accepted, since black sometimes looks dull or grayish under water
- Added shape checks so that only rectangular looking shapes are accepted as posts; Implemented to avoid random black objects as being registered as the gate 
- Added height checks for the blobs found; That way only blobs that are resting at around the same height would be taken into consideration for a gate; Tradeoff: Can't detect partial gates 
- Added a pattern check so that posts that are only half black and red are accepted; Also checked that the the possible pair post had it's color order swapped to eliminate capturing the posts' reflections
- Lowered the confidence cutoff

YOLO:
- Trained on 1,467 images
- Learns what a gate looks like from examples, so it doesn't depend on color rules or both posts being visible

## Training Arguments
- Epochs: 30
- Image Size: 640 px
- Batch Size: 16
- Device: mps
- Optimizer: AdamW
- Seed: 0

## Results Table

| Metric | Classical | YOLO |
|---|---|---|
| Precision | 1.00 | 0.936 |
| Recall | 0.086 | 0.936 |
| mAP50 | 0.089 | 0.906 |
| Mean center error | 0.0064 | 0.0065 |
| FPS | 796 | 28 |
| Recall, truncated | 0.00 | 0.93 |
| Recall, whole | 0.17 | 0.94 |

## Limitations and Issues
- Data leakage: 301 of 611 original photos have augmented copies in more than one split, so YOLO's val and test scores are likely inflated. Classical is unaffected because it isn't trained.
- Classical is very precise (1.00) but finds only 12 of 140 test gates (recall 0.086); it can't detect any of the 71 cut-off gates because it needs both posts visible
- Classical struggles on washed-out photos where the red isn't as distinguishable
- In the YOLO failures most were zoomed-in photos where the gate fills the frame
- The scored false positive count doesn't match the false alarms visible when I manually reviewed the annotated images


