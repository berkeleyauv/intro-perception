import csv
import sys

import cv2 as cv
import numpy as np
import pytest
from gate_helpers import write_labeled_image

from intro_perception import data_tools
from intro_perception.data_tools import (
    annotate_directory,
    parse_label,
    split_dataset,
    write_label,
)

def label_file(tmp_path, text):
    path = tmp_path / "x.txt"
    path.write_text(text)
    return path

def make_raw(tmp_path, count=10, **kwargs):
    raw = tmp_path / "raw"
    for index in range(count):
        write_labeled_image(raw, f"img_{index:03d}", **kwargs)
    return raw

def make_separate(tmp_path, count=10, unlabeled=0):
    """Images in one folder, labels in another (like the labeling pipeline's output)."""
    images, labels = tmp_path / "images", tmp_path / "labels_detect"
    for index in range(count):
        write_labeled_image(images, f"img_{index:03d}", labels_dir=labels)
    for index in range(unlabeled):  # an image the labeling pipeline skipped: no label written
        write_labeled_image(images, f"skipped_{index}", labels_dir=tmp_path / "scratch")
    return images, labels

# --- Parse Label Tests ---

def test_empty_label_means_no_gate(tmp_path):
    assert parse_label(label_file(tmp_path, "")) is None
    assert parse_label(label_file(tmp_path, "\n  \n")) is None

def test_valid_label_is_parsed(tmp_path):
    assert parse_label(label_file(tmp_path, "0 0.5 0.4 0.3 0.2\n")) == (0.5, 0.4, 0.3, 0.2)

@pytest.mark.parametrize(
    "text, message",
    [
        ("0 0.5 0.5 0.4 0.6\n0 0.3 0.3 0.1 0.1\n", "one gate box"),
        ("0 0.5 0.5 0.4\n", "class cx cy w h"),
        ("a 0.5 0.5 0.4 0.6\n", "non-numeric"),
        ("0 1.5 0.5 0.4 0.6\n", "0-1"),
        ("0 0.5 0.5 0 0.6\n", "positive size"),
    ],
)
def test_malformed_labels_fail_loudly(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        parse_label(label_file(tmp_path, text))

@pytest.mark.parametrize("class_id", [1, 2, 3])
def test_only_class_zero_is_accepted(tmp_path, class_id):
    # Labels from the old 3-orientation-class dataset must not be read as gates silently.
    with pytest.raises(ValueError, match="only class is 'gate'"):
        parse_label(label_file(tmp_path, f"{class_id} 0.5 0.5 0.4 0.6\n"))

def test_write_label_round_trips(tmp_path):
    path = tmp_path / "x.txt"
    write_label(path, (20, 10, 100, 80), (100, 200, 3))
    assert path.read_text().split()[0] == "0"
    assert parse_label(path) == pytest.approx((0.35, 0.5, 0.5, 0.8))

# --- Split Dataset Tests (labels next to images) ---

def test_split_writes_layout_labels_and_yaml(tmp_path):
    raw, out = make_raw(tmp_path, 10), tmp_path / "generated"
    summary = split_dataset(raw, out)
    assert {split: counts["images"] for split, counts in summary.items()} == {
        "train": 8, "val": 1, "test": 1,
    }
    for split, counts in summary.items():
        images = sorted((out / split / "images").iterdir())
        labels = sorted((out / split / "labels").iterdir())
        assert len(images) == len(labels) == counts["images"]
        assert [i.stem for i in images] == [label.stem for label in labels]
    yaml_text = (out / "dataset.yaml").read_text()
    assert "train: train/images" in yaml_text and "test: test/images" in yaml_text
    assert "0: gate" in yaml_text and "gate_left" not in yaml_text
    # `path:` would be resolved against the working directory by Ultralytics.
    assert "path:" not in yaml_text

def test_split_is_reproducible_and_seeded(tmp_path):
    raw = make_raw(tmp_path, 12)
    split_dataset(raw, tmp_path / "a", seed=1)
    split_dataset(raw, tmp_path / "b", seed=1)
    split_dataset(raw, tmp_path / "c", seed=2)

    def test_names(root):
        return sorted(p.name for p in (root / "test" / "images").iterdir())

    assert test_names(tmp_path / "a") == test_names(tmp_path / "b")
    assert test_names(tmp_path / "a") != test_names(tmp_path / "c")

@pytest.mark.parametrize("count", [3, 4, 5, 8])
def test_small_datasets_still_get_nonempty_val_and_test(tmp_path, count):
    summary = split_dataset(make_raw(tmp_path, count), tmp_path / "out")
    assert all(counts["images"] >= 1 for counts in summary.values())
    assert sum(counts["images"] for counts in summary.values()) == count

def test_too_few_images_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="at least 3"):
        split_dataset(make_raw(tmp_path, 2), tmp_path / "out")

def test_split_counts_gate_and_no_gate_images(tmp_path):
    raw = tmp_path / "raw"
    for index in range(6):
        write_labeled_image(raw, f"gate_{index}")
    for index in range(2):
        write_labeled_image(raw, f"empty_{index}", gate=False)
    summary = split_dataset(raw, tmp_path / "out")
    assert sum(counts["no_gate"] for counts in summary.values()) == 2
    assert sum(counts["gate"] for counts in summary.values()) == 6

def test_split_accepts_jpg_jpeg_png_in_any_case(tmp_path):
    raw = tmp_path / "raw"
    for name, ext in (("a", ".jpg"), ("b", ".jpeg"), ("c", ".png"), ("d", ".JPG"), ("e", ".PNG")):
        write_labeled_image(raw, name, ext=ext)
    summary = split_dataset(raw, tmp_path / "out")
    assert sum(counts["images"] for counts in summary.values()) == 5

def test_split_refuses_unlabeled_images(tmp_path):
    raw = make_raw(tmp_path, 5)
    (raw / "img_002.txt").unlink()
    with pytest.raises(ValueError, match="img_002.png"):
        split_dataset(raw, tmp_path / "out")

def test_split_rejects_json_only_annotations(tmp_path):
    # The old annotation tool wrote .json sidecars; they must not pass for labels.
    raw = make_raw(tmp_path, 5)
    for text_label in list(raw.glob("*.txt")):
        text_label.with_suffix(".json").write_text("{}")
        text_label.unlink()
    with pytest.raises(ValueError, match="no .txt label"):
        split_dataset(raw, tmp_path / "out")

def test_split_rejects_malformed_label_files(tmp_path):
    raw = make_raw(tmp_path, 5)
    (raw / "img_001.txt").write_text("0 0.5 0.5 0.4 0.6\n0 0.1 0.1 0.1 0.1\n")
    with pytest.raises(ValueError, match="img_001.txt"):
        split_dataset(raw, tmp_path / "out")

def test_split_rejects_old_three_class_labels(tmp_path):
    raw = make_raw(tmp_path, 5)
    (raw / "img_001.txt").write_text("2 0.5 0.5 0.4 0.6\n")
    with pytest.raises(ValueError, match="only class is 'gate'"):
        split_dataset(raw, tmp_path / "out")

def test_split_ignores_videos_and_subfolders(tmp_path):
    raw = make_raw(tmp_path, 5)
    (raw / "clip.mp4").write_bytes(b"not really a video")
    (raw / "excluded").mkdir()
    write_labeled_image(raw / "excluded", "skipped")
    summary = split_dataset(raw, tmp_path / "out")
    assert sum(counts["images"] for counts in summary.values()) == 5

def test_split_will_not_mix_with_an_earlier_split_unless_asked(tmp_path):
    raw, out = make_raw(tmp_path, 6), tmp_path / "out"
    split_dataset(raw, out, seed=1)
    with pytest.raises(ValueError, match="--overwrite"):
        split_dataset(raw, out, seed=2)
    split_dataset(raw, out, seed=2, overwrite=True)
    names = [p.name for split in ("train", "val", "test") for p in (out / split / "images").iterdir()]
    assert len(names) == len(set(names)) == 6  # no image appears in two splits

# --- Split Dataset Tests (labels in a separate folder) ---

def test_split_reads_labels_from_a_separate_folder(tmp_path):
    images, labels = make_separate(tmp_path, 10)
    out = tmp_path / "generated"
    summary = split_dataset(images, out, labels_dir=labels)
    assert sum(counts["images"] for counts in summary.values()) == 10
    assert not list(images.glob("*.txt"))           # the image folder was never touched
    for split in ("train", "val", "test"):
        assert {p.stem for p in (out / split / "images").iterdir()} == {
            p.stem for p in (out / split / "labels").iterdir()
        }

def test_images_without_a_label_are_left_out_and_reported(tmp_path, capsys):
    images, labels = make_separate(tmp_path, 8, unlabeled=2)
    summary = split_dataset(images, tmp_path / "out", labels_dir=labels)
    assert sum(counts["images"] for counts in summary.values()) == 8
    assert "2 image(s) have no label and were left out" in capsys.readouterr().out
    used = {p.name for s in ("train", "val", "test") for p in (tmp_path / "out" / s / "images").iterdir()}
    assert not any(name.startswith("skipped_") for name in used)

def test_a_label_without_an_image_is_an_error(tmp_path):
    images, labels = make_separate(tmp_path, 6)
    (images / "img_003.png").unlink()
    with pytest.raises(ValueError, match="no matching image.*img_003"):
        split_dataset(images, tmp_path / "out", labels_dir=labels)

def test_an_empty_labels_folder_is_an_error(tmp_path):
    images, _ = make_separate(tmp_path, 4)
    empty = tmp_path / "empty_labels"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .txt labels"):
        split_dataset(images, tmp_path / "out", labels_dir=empty)

def test_separate_labels_keep_empty_files_as_no_gate_images(tmp_path):
    images, labels = make_separate(tmp_path, 6)
    write_labeled_image(images, "negative", gate=False, labels_dir=labels)
    summary = split_dataset(images, tmp_path / "out", labels_dir=labels)
    assert sum(counts["no_gate"] for counts in summary.values()) == 1

def write_manifest(labels, rows, fields=("filename", "status", "reason", "truncated")):
    with (labels / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def test_manifest_flags_flow_into_a_split_manifest(tmp_path):
    images, labels = make_separate(tmp_path, 10)
    rows = [
        {"filename": f"img_{i:03d}.png", "status": "exported", "reason": "", "truncated": i < 3}
        for i in range(10)
    ]
    write_manifest(labels, rows)
    out = tmp_path / "out"
    summary = split_dataset(images, out, labels_dir=labels)
    assert sum(counts["truncated"] for counts in summary.values()) == 3
    with (out / "manifest.csv").open(newline="") as stream:
        written = list(csv.DictReader(stream))
    assert len(written) == 10 and {row["split"] for row in written} == {"train", "val", "test"}
    assert sum(row["truncated"] == "True" for row in written) == 3
    assert list(written[0]) == ["filename", "split", "status", "truncated"]

def test_no_manifest_means_no_truncation_columns(tmp_path):
    images, labels = make_separate(tmp_path, 6)
    out = tmp_path / "out"
    summary = split_dataset(images, out, labels_dir=labels)
    assert all("truncated" not in counts for counts in summary.values())
    assert not (out / "manifest.csv").exists()

def test_a_manifest_without_a_filename_column_is_rejected(tmp_path):
    images, labels = make_separate(tmp_path, 6)
    write_manifest(labels, [{"status": "exported"}], fields=("status",))
    with pytest.raises(ValueError, match="'filename' column"):
        split_dataset(images, tmp_path / "out", labels_dir=labels)

def test_cli_split_accepts_a_labels_folder(tmp_path, monkeypatch, capsys):
    images, labels = make_separate(tmp_path, 10)
    out = tmp_path / "generated"
    monkeypatch.setattr(
        sys, "argv",
        ["data", "split", "--data", str(images), "--labels", str(labels), "--output", str(out)],
    )
    data_tools.main()
    printed = capsys.readouterr().out
    assert "train: images=8, gate=8, no_gate=0" in printed
    assert (out / "dataset.yaml").exists()

# -- Test Annotate ---

def test_annotate_writes_labels_skips_done_and_moves_excluded(tmp_path, monkeypatch):
    images = tmp_path / "raw"
    images.mkdir()
    for name in ("a", "b", "c", "d", "e"):
        cv.imwrite(str(images / f"{name}.png"), np.zeros((100, 200, 3), np.uint8))
    (images / "d.txt").write_text("0 0.5 0.5 0.2 0.2\n")  # already labeled: must be left alone

    boxes = iter([(20, 10, 100, 80), (0, 0, 0, 0), (20, 10, 100, 80), (20, 10, 100, 80)])
    answers = iter(["", "s", ""])  # a=save, c=skip, e=save
    prompts = []
    monkeypatch.setattr(data_tools.cv, "selectROI", lambda *args, **kwargs: next(boxes))
    monkeypatch.setattr(data_tools.cv, "destroyWindow", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(answers))

    annotate_directory(images)

    assert parse_label(images / "a.txt") == pytest.approx((0.35, 0.5, 0.5, 0.8))
    assert parse_label(images / "b.txt") is None  # cancelled box -> explicit no-gate label
    assert not (images / "c.png").exists() and (images / "excluded" / "c.png").exists()
    assert (images / "d.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert parse_label(images / "e.txt") == pytest.approx((0.35, 0.5, 0.5, 0.8))
    assert not any("orientation" in prompt for prompt in prompts)  # no l/h/r question any more