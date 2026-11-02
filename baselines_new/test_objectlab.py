"""论文公式与 anno/image 适配的回归检查：python -B -m unittest baselines_new.test_objectlab。"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines_new.objectLab import (
    box_matrices, build_ranking, image_miss_score, objectlab_score, softmin,
)


def prediction(box, category=0, conf=0.99, box_id=0):
    return {"bbox": box, "predicted_cls": category, "conf": conf,
            "predicted_box_id": box_id, "probs": [0.99, 0.01]}


class ObjectLabTests(unittest.TestCase):
    def setUp(self):
        self.annos = [("0", 0, [0, 0, 10, 10])]

    def test_similarity_formula_and_resize_invariance(self):
        predictions = [prediction([0, 0, 5, 10]), prediction([20, 20, 30, 30])]
        iou, sim = box_matrices(self.annos, predictions, (100, 100))
        self.assertAlmostEqual(iou[0, 0], 0.5)
        self.assertAlmostEqual(sim[0, 0], 0.45 + 0.1 * math.exp(-0.05 / 0.1))
        self.assertEqual(iou[0, 1], 0)
        self.assertGreater(sim[0, 1], 0)
        scaled_annos = [("0", 0, [0, 0, 20, 20])]
        scaled_preds = [prediction([2 * x for x in pred["bbox"]]) for pred in predictions]
        iou2, sim2 = box_matrices(scaled_annos, scaled_preds, (200, 200))
        np.testing.assert_allclose(iou, iou2)
        np.testing.assert_allclose(sim, sim2)

    def test_swapped_threshold_is_confidence_not_iou(self):
        low = [prediction([0, 0, 10, 10], category=1, conf=0.95)]
        iou, sim = box_matrices(self.annos, low, (100, 100))
        self.assertEqual(objectlab_score(0, low, iou[0], sim[0]), 1)
        high = [prediction([0, 0, 5, 10], category=1)]
        iou, sim = box_matrices(self.annos, high, (100, 100))
        self.assertAlmostEqual(objectlab_score(0, high, iou[0], sim[0]), math.sqrt(1 - sim[0, 0]))

    def test_no_predictions_and_no_same_class_overlap(self):
        iou, sim = box_matrices(self.annos, [], (100, 100))
        self.assertEqual(objectlab_score(0, [], iou[0], sim[0]), 1)
        self.assertEqual(image_miss_score(self.annos, [], iou, sim, 0.01), 1)
        preds = [prediction([20, 20, 30, 30])]
        iou, sim = box_matrices(self.annos, preds, (100, 100))
        self.assertEqual(objectlab_score(0, preds, iou[0], sim[0]), 1)

    def test_any_class_overlap_suppresses_miss(self):
        annos = self.annos + [("1", 0, [50, 50, 60, 60])]
        for category in (0, 1):
            preds = [prediction([0, 0, 10, 10], category=category)]
            iou, sim = box_matrices(annos, preds, (100, 100))
            self.assertEqual(image_miss_score(annos, preds, iou, sim, 0.01), 1)

    def test_miss_similarity_and_absent_class(self):
        preds = [prediction([20, 20, 30, 30])]
        iou, sim = box_matrices(self.annos, preds, (100, 100))
        score = image_miss_score(self.annos, preds, iou, sim, 0.001)
        self.assertAlmostEqual(score, sim[0, 0])
        preds[0]["predicted_cls"] = 1
        score = image_miss_score(self.annos, preds, iou, sim, 0.001)
        self.assertAlmostEqual(score, 0.001 * (1 - 0.99))
        preds[0]["conf"] = 0.95
        self.assertEqual(image_miss_score(self.annos, preds, iou, sim, 0.001), 1)

    def test_softmin_is_paper_weighted_average(self):
        expected = (0.2 * math.exp(0.8) + 0.8 * math.exp(0.2)) / (math.exp(0.8) + math.exp(0.2))
        self.assertAlmostEqual(softmin([0.2, 0.8]), expected)
        self.assertEqual(softmin([]), 1)

    def test_end_to_end_global_min_filter_and_image_aggregation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = {"images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100}
                               for i in range(3)],
                    "annotations": [{"id": 7, "image_id": 0, "category_id": 0, "bbox": [0, 0, 10, 10]}]}
            nearby = prediction([0, 0, 5, 10])
            far = prediction([80, 80, 90, 90], category=1, box_id=1)
            # conf=0.5 的异类完美匹配应被全局过滤，不改变 sim* 或类别分。
            ignored = prediction([0, 0, 10, 10], category=1, conf=0.5, box_id=2)
            predictions = {"0.jpg": {"predicted_bboxs": [nearby, far, ignored]},
                           "1.jpg": {"predicted_bboxs": [prediction([0, 0, 10, 10])]}}
            (root / "annos.json").write_text(json.dumps(data))
            (root / "preds.json").write_text(json.dumps(predictions))
            rows = build_ranking(root / "annos.json", root / "preds.json")
            self.assertEqual(len(rows), 4)
            _, sim = box_matrices(self.annos, [nearby, far], (100, 100))
            empty_iou, empty_sim = box_matrices([], [nearby], (100, 100))
            empty_score = image_miss_score([], [nearby], empty_iou, empty_sim, float(sim.min()))
            self.assertAlmostEqual(empty_score, sim.min() * (1 - 0.99))
            # 两张漏标图同分时保持插入顺序；anno 随后，无预测图片最后。
            self.assertEqual(rows, ["0.jpg", "1.jpg", 7, "2.jpg"])
            self.assertIs(type(rows[2]), int)


if __name__ == "__main__":
    unittest.main()
