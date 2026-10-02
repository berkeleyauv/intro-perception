import csv
import sys

import pytest
from gate_helpers import write_labeled_image

from intro_perception import evaluate
from intro_perception.evaluate import (
    box_iou,
    load_truncated,
    score_by_truncation,
    evaluate_test_set,
    load_records,
    load_truth,
    score_records,
    warn_about_truth,
)


def visible(**overrides):
    box = {
        "visible": True,
        "confidence": 0.9,
        "center_x": 0.5,
        "center_y": 0.5,
        "box_x": 0.2,
        "box_y": 0.2,
        "box_width": 0.6,
        "box_height": 0.6,
    }
    box.update(overrides)
    return box


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


def test_scores_have_no_orientation_metrics():
    score = score_records({0: visible()}, {0: visible()})
    assert set(score) == {
        "frames", "precision_iou50", "recall_iou50", "map50", "mean_center_error", "fps",
    }


def test_missed_gate_and_false_alarm_score_zero():
    truth = {0: visible(), 1: {"visible": False}}
    predictions = {0: {"visible": False}, 1: visible()}
    score = score_records(truth, predictions)
    assert score["precision_iou50"] == 0.0
    assert score["recall_iou50"] == 0.0


def test_a_box_in_the_wrong_place_is_both_a_miss_and_a_false_positive():
    shifted = visible(box_x=0.9, box_width=0.1, center_x=0.95)
    score = score_records({0: visible()}, {0: shifted})
    assert score["recall_iou50"] == 0.0 and score["precision_iou50"] == 0.0


def test_map_depends_on_ranking_confident_correct_boxes_first():
    truth = {0: visible(), 1: {"visible": False}}
    good = score_records(truth, {0: visible(confidence=0.95), 1: visible(confidence=0.5)})
    bad = score_records(truth, {0: visible(confidence=0.5), 1: visible(confidence=0.95)})
    assert good["map50"] == 1.0
    assert bad["map50"] == pytest.approx(0.5)


def make_test_dir(tmp_path):
    test_dir = tmp_path / "test"
    for index, shift in enumerate((0, 10, -10)):
        write_labeled_image(
            test_dir / "images", f"gate_{index}", shift=shift, labels_dir=test_dir / "labels"
        )
    write_labeled_image(test_dir / "images", "empty", gate=False, labels_dir=test_dir / "labels")
    return test_dir


def test_load_truth_reads_boxes_and_no_gate_images(tmp_path):
    truth = load_truth(make_test_dir(tmp_path))
    assert set(truth) == {"gate_0", "gate_1", "gate_2", "empty"}
    assert truth["empty"] == {"visible": False}
    gate = truth["gate_0"]
    assert gate["visible"] and "orientation" not in gate
    assert gate["box_x"] == pytest.approx(gate["center_x"] - gate["box_width"] / 2)


def test_load_truth_accepts_png_and_jpg_and_rejects_unlabeled(tmp_path):
    test_dir = tmp_path / "test"
    for name, ext in (("a", ".png"), ("b", ".jpg"), ("c", ".jpeg")):
        write_labeled_image(test_dir / "images", name, ext=ext, labels_dir=test_dir / "labels")
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


def test_warns_when_test_set_has_no_gates(capsys):
    warn_about_truth({"a": {"visible": False}, "b": {"visible": False}})
    assert "no gate images" in capsys.readouterr().err
    warn_about_truth({"a": {"visible": True}})
    assert capsys.readouterr().err == ""


# --- cut-off gates scored separately ---

def write_split_manifest(path, truncated_ids, all_ids):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "split", "status", "truncated"])
        writer.writeheader()
        for frame_id in all_ids:
            writer.writerow({
                "filename": f"{frame_id}.png", "split": "test", "status": "gate",
                "truncated": frame_id in truncated_ids,
            })


def test_load_truncated_reads_the_flag(tmp_path):
    write_split_manifest(tmp_path / "manifest.csv", {"b"}, ["a", "b", "c"])
    assert load_truncated(tmp_path / "manifest.csv") == {"b"}


def test_cut_off_gates_are_scored_separately():
    truth = {"whole": visible(), "cut": visible(), "empty": {"visible": False}}
    predictions = {"whole": visible(), "cut": {"visible": False}, "empty": {"visible": False}}
    groups = score_by_truncation(truth, predictions, {"cut"})
    assert groups["whole"]["recall_iou50"] == 1.0 and groups["whole"]["frames"] == 2
    assert groups["truncated"]["recall_iou50"] == 0.0 and groups["truncated"]["frames"] == 1


def test_a_group_with_no_frames_is_omitted():
    groups = score_by_truncation({"a": visible()}, {"a": visible()}, set())
    assert list(groups) == ["whole"]


def test_live_evaluation_adds_the_breakdown_only_when_a_manifest_exists(tmp_path):
    test_dir = make_test_dir(tmp_path)
    assert "by_truncation" not in evaluate_test_set(test_dir, method="classical")["classical"]
    write_split_manifest(tmp_path / "manifest.csv", {"gate_1"}, ["gate_0", "gate_1", "gate_2", "empty"])
    scores = evaluate_test_set(test_dir, method="classical")["classical"]
    assert set(scores["by_truncation"]) == {"whole", "truncated"}
    assert scores["by_truncation"]["truncated"]["frames"] == 1
    assert scores["by_truncation"]["whole"]["frames"] == 3