import json

import cv2 as cv
import numpy as np
import pytest

from intro_perception.data_tools import (
    build_dataset,
    evenly_spaced_indices,
    split_sources,
    validate_annotation,
)


def test_evenly_spaced_indices_include_ends():
    assert evenly_spaced_indices(10, 3) == [0, 4, 9]


def test_split_sources_keeps_each_video_in_one_split():
    splits = split_sources(["a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"])
    assert set.union(*splits.values()) == {"a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"}
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"])


def test_minimum_three_videos_gives_three_nonempty_splits():
    splits = split_sources(["a.mp4", "b.mp4", "c.mp4"])
    assert all(splits.values())


def test_validation_rejects_unfinished_frame():
    with pytest.raises(ValueError, match="not been labeled"):
        validate_annotation(
            {"source_video": "a.mp4", "frame_index": 0, "visible": None},
            (100, 200, 3),
        )


def test_build_writes_yolo_labels_and_truth(tmp_path):
    images = tmp_path / "working"
    images.mkdir()
    for index, source in enumerate(("a.mp4", "b.mp4", "c.mp4")):
        image = images / f"frame_{index}.jpg"
        cv.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
        annotation = {
            "source_video": source,
            "frame_index": 10,
            "visible": True,
            "orientation": "gate_head_on",
            "bbox_xywh_pixels": [20, 10, 100, 80],
        }
        image.with_suffix(".json").write_text(json.dumps(annotation))
    output = tmp_path / "generated"
    manifest = build_dataset(images, output)
    assert len(manifest) == 3
    assert (output / "dataset.yaml").is_file()
    assert (output / "truth_test.jsonl").read_text().strip()
