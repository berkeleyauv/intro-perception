import json
import sys

import cv2 as cv
import numpy as np
import pytest
from gate_helpers import make_gate_frame

from intro_perception import run, visualize
from intro_perception.images import list_images


def write_image(path, frame=None):
    frame = np.zeros((20, 30, 3), np.uint8) if frame is None else frame
    assert cv.imwrite(str(path), frame)
    return path


# --- image discovery -----------------------------------------------------------


def test_list_images_is_sorted_case_insensitive_and_non_recursive(tmp_path):
    for name in ("b.png", "a.jpg", "C.JPEG", "D.PNG"):
        write_image(tmp_path / name)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "clip.mp4").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    write_image(tmp_path / "sub" / "z.png")
    assert [p.name for p in list_images(tmp_path)] == ["C.JPEG", "D.PNG", "a.jpg", "b.png"]


def test_list_images_rejects_videos_and_missing_paths(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="videos are not"):
        list_images(video)
    with pytest.raises(ValueError, match="does not exist"):
        list_images(tmp_path / "nope")


def test_list_images_rejects_empty_folder_and_name_clashes(tmp_path):
    (tmp_path / "only_video.mp4").write_bytes(b"x")
    with pytest.raises(ValueError, match="no images"):
        list_images(tmp_path)
    write_image(tmp_path / "a.jpg")
    write_image(tmp_path / "a.png")
    with pytest.raises(ValueError, match="share the name 'a'"):
        list_images(tmp_path)


# --- run.py --------------------------------------------------------------------


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run", *args])
    run.main()


def test_run_writes_predictions_and_annotated_images(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        write_image(images / f"gate_{index}.png", make_gate_frame(shift=index))
    out = tmp_path / "out"
    run_cli(monkeypatch, "--data", str(images), "--method", "classical", "--output", str(out))

    records = [
        json.loads(line) for line in (out / "predictions_classical.jsonl").read_text().splitlines()
    ]
    assert [r["frame_id"] for r in records] == ["gate_0", "gate_1", "gate_2"]
    assert all(r["visible"] and r["inference_sec"] > 0 for r in records)
    assert all("source_video" not in r and "frame_index" not in r for r in records)
    assert sorted(p.name for p in (out / "annotated").iterdir()) == [
        "gate_0.jpg", "gate_1.jpg", "gate_2.jpg",
    ]


def test_run_rejects_video_input_before_writing_anything(tmp_path, monkeypatch):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="videos are not"):
        run_cli(monkeypatch, "--data", str(video), "--method", "classical", "--output", str(out))
    assert not out.exists()


# --- visualize.py ----------------------------------------------------------------


def test_visualize_imports_and_has_registry():
    # Regression: a merged import line once turned `registry` into a comment.
    assert visualize.registry is not None


def test_visualize_data_sources_are_images_only(tmp_path):
    write_image(tmp_path / "a.png")
    write_image(tmp_path / "b.JPG")
    assert [p.split("/")[-1] for p in visualize.data_sources(tmp_path)] == ["a.png", "b.JPG"]

    write_image(tmp_path / "c.jpeg")  # the production FrameWrapper cannot read .jpeg
    with pytest.raises(ValueError, match="only reads .jpg, .png"):
        visualize.data_sources(tmp_path)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ValueError, match="videos are not"):
        visualize.data_sources(video)
    with pytest.raises(ValueError):  # the old webcam source is gone too
        visualize.data_sources("webcam")


def test_visualize_main_passes_images_to_the_production_run(tmp_path, monkeypatch):
    write_image(tmp_path / "a.png")
    seen = {}
    monkeypatch.setattr(visualize, "run", lambda sources, algorithm, **kwargs: seen.update(
        sources=sources, algorithm=algorithm, **kwargs))
    monkeypatch.setattr(sys, "argv", ["vis", "--data", str(tmp_path)])
    visualize.main()
    assert seen["sources"] == [str(tmp_path / "a.png")]
    assert type(seen["algorithm"]).__name__ == "ClassicalGatePerceiver"
    assert seen["save_video"] is False