from intro_perception.evaluate import box_iou, score_records


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
