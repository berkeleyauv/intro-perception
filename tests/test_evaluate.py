from intro_perception.evaluate import score_records


def test_perfect_predictions_score_one():
    truth = {
        0: {"visible": True, "center_x": 0.4, "center_y": 0.5},
        1: {"visible": False},
    }
    predictions = {
        0: {"visible": True, "center_x": 0.4, "center_y": 0.5},
        1: {"visible": False},
    }
    score = score_records(truth, predictions)
    assert score["visibility_accuracy"] == 1.0
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["mean_center_error"] == 0.0
