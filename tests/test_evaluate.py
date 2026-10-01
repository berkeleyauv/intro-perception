import sys

import pytest
from intro_perception.gate_helpers import write_labeled_image

from intro_perception import evaluate
from intro_perception.evaluate import (
    box_iou,
    evaluate_test_set,
    load_records,
    load_truth,
    score_records,
    warn_about_truth,
)

def visible(orientation="head_on"):
    return {
        "visible": True,
        "confidence": 0.9,
        "center_x": 0.5,
        "center_y": 0.5,
        "orientation": orientation,
        "box_x": 0.2,
        "box_y": 0.2,
        "box_width": 0.6,
        "box_height": 0.6,
    }


def test_identical_boxes_have_one_iou():
    assert box_iou(visible(), visible()) == 1.0


def test_perfect_predictions_score_one():
    truth = {0: visible(), 1: {"visible": False}}
    predictions = {0: visible(), 1: {"visible": False}}
    score = score_records(truth, predictions)
    assert score["precision_iou50"] == 1.0
    assert score["recall_iou50"] == 1.0
    assert score["map50"] == 1.0
    assert score["mean_center_error"] == 0.0
    assert score["orientation_macro_f1"] == 1.0


def test_wrong_orientation_lowers_macro_f1():
    score = score_records({0: visible("left")}, {0: visible("right")})
    assert score["orientation_macro_f1"] == 0.0

def make_test_dir(tmp_path):
    test_dir = tmp_path / "test"
    for index, shift in enumerate((0, 10, -10)):
        write_labeled_image(test_dir / "images", f"gate_{index}", shift=shift)
    write_labeled_image(test_dir / "images", "empty", gate=False)
    # labels live next to images in the helper, so move them into labels/
    (test_dir / "labels").mkdir()
    for label in (test_dir / "images").glob("*.txt"):
        label.rename(test_dir / "labels" / label.name)
    return test_dir

def test_load_truth_reads_boxes_and_no_gate_images(tmp_path):
    truth = load_truth(make_test_dir(tmp_path))
    assert set(truth) == {"gate_0", "gate_1", "gate_2", "empty"}
    assert truth["empty"] == {"visible": False}
    gate = truth["gate_0"]
    assert gate["visible"] and gate["orientation"] == "head_on"
    assert gate["box_x"] == pytest.approx(gate["center_x"] - gate["box_width"] / 2)

def test_load_truth_accepts_png_and_jpg_and_rejects_unlabeled(tmp_path):
    test_dir = tmp_path / "test"
    for name, ext in (("a", ".png"), ("b", ".jpg"), ("c", ".jpeg")):
        write_labeled_image(test_dir / "images", name, ext=ext)
    (test_dir / "labels").mkdir()
    for label in (test_dir / "images").glob("*.txt"):
        label.rename(test_dir / "labels" / label.name)
    assert set(load_truth(test_dir)) == {"a", "b", "c"}
    (test_dir / "labels" / "b.txt").unlink()
    with pytest.raises(ValueError, match="missing label for b.jpg"):
        load_truth(test_dir)
 
def test_load_records_is_keyed_by_frame_id(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"frame_id": "a", "visible": false}\n\n{"frame_id": "b", "visible": false}\n')
    assert set(load_records(path)) == {"a", "b"}
    path.write_text('{"visible": false}\n')
    with pytest.raises(ValueError, match="missing frame_id"):
        load_records(path)

def test_live_evaluation_scores_classical_and_measures_speed(tmp_path):
    scores = evaluate_test_set(make_test_dir(tmp_path), method="classical")
    classical = scores["classical"]
    assert classical["frames"] == 4
    assert classical["precision_iou50"] == 1.0 and classical["recall_iou50"] == 1.0
    assert classical["orientation_macro_f1"] == 1.0
    assert classical["fps"] is not None and classical["fps"] > 0

def test_cli_refuses_predictions_from_a_different_test_set(tmp_path, monkeypatch):
    test_dir = make_test_dir(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"frame_id": "gate_0", "visible": false}\n')
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate", "--test-dir", str(test_dir), "--predictions", str(predictions)],
    )
    with pytest.raises(SystemExit, match="no prediction for 3 test image"):
        evaluate.main()

def test_warns_when_test_set_cannot_support_metrics(capsys):
    warn_about_truth({"a": {"visible": False}, "b": {"visible": False}})
    assert "no gate images" in capsys.readouterr().err
    warn_about_truth({"a": {"visible": True, "orientation": "head_on"}})
    err = capsys.readouterr().err
    assert "no 'left' gates" in err and "no 'right' gates" in err and "head_on" not in err