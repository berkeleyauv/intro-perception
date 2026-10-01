# Local data layout

Mentor-supplied images can live anywhere outside Git; copy the ones you are
working with into `data/raw/`. The project tools create:

```text
data/raw/        your images plus a YOLO .txt label next to each (see below)
data/generated/  the train/val/test split and dataset.yaml
    train/images, train/labels
    val/images,   val/labels
    test/images,  test/labels
```

Only still images (`.jpg`, `.jpeg`, `.png`) are used; videos are ignored or
rejected. Image names must be unique once the extension is dropped (`a.jpg` and
`a.png` clash), because the name is the key for labels and predictions.

A label file holds one line, `class cx cy w h`, normalized to 0-1, with class
0/1/2 = `gate_left` / `gate_head_on` / `gate_right`. An empty `.txt` means the
image has no gate. `intro-perception-data annotate` writes these for you;
ambiguous images you skip are moved to `data/raw/excluded/`.

`data/raw/` and `data/generated/` are ignored by Git. To share labels within a
student team, archive them outside the repository or use an approved team
storage location. Public starter samples may be committed under
`data/samples/`; private staff test images must never be committed.