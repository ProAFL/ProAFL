"""运行：python -B -m unittest baselines_new.test_others。"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from baselines_new import others


class OtherBaselineTests(unittest.TestCase):
    def test_probability_scores(self):
        probs = [0.5, 0.5, 0.0]
        self.assertAlmostEqual(others.calcu_entropy(probs), math.log(2))
        self.assertEqual(others.caclu_deepgini(probs), 0.5)
        self.assertEqual(others.caclu_margin(probs), 1)
        self.assertEqual(probs, [0.5, 0.5, 0.0])
        self.assertEqual(others.calcu_entropy([1, 0]), 0)

    def test_loss_uses_annotation_probability_and_smooth_l1(self):
        score = others.caclu_loss([0.8, 0.2], 1, [0, 0, 0.5, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(score, -math.log(0.2) + 0.5 * 0.5 ** 2 / 4)
        self.assertTrue(math.isfinite(others.caclu_loss([1, 0], 1, [0]*4, [0]*4)))

    def test_matching_ranking_and_empty_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = {
                "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100}
                           for i in range(4)],
                "annotations": [{"id": i, "image_id": i, "category_id": 0, "bbox": [0, 0, 10, 10]}
                                for i in range(3)],
            }
            def pred(box_id, probs, conf, bbox=None):
                return {"predicted_box_id": box_id, "predicted_cls": 1,
                        "bbox": bbox or [0, 0, 10, 10], "conf": conf, "probs": probs}
            # 最大 IoU 优先于置信度；并列比较 conf，再选择最小 ID。
            choices = [pred(1, [1, 0], 0.99, [0, 0, 5, 10]),
                       pred(2, [1, 0], 0.8), pred(4, [1, 0], 0.9),
                       pred(3, [0.5, 0.5], 0.9)]
            predictions = {"0.jpg": {"predicted_bboxs": choices},
                           "1.jpg": {"predicted_bboxs": [pred(0, [0.9, 0.1], 0.9)]},
                           "2.jpg": {"predicted_bboxs": [pred(0, [1, 0], 1, [20, 20, 30, 30])]}}
            anno_path, pred_path = root / "annos.json", root / "preds.json"
            anno_path.write_text(json.dumps(data))
            pred_path.write_text(json.dumps(predictions))
            expected = [2, 0, 1, "0.jpg", "1.jpg", "2.jpg", "3.jpg"]
            for baseline in others.BASELINES:
                with self.subTest(baseline=baseline):
                    self.assertEqual(others.build_ranking(anno_path, pred_path, baseline, mix_images=False), expected)
                    mixed = others.build_ranking(anno_path, pred_path, baseline, mix_images=True)
                    self.assertEqual(mixed, [2, 0, "2.jpg", 1, "0.jpg", "1.jpg", "3.jpg"])
                    self.assertEqual([item for item in mixed if isinstance(item, int)], [2, 0, 1])
            # loss 在图片与框同比放大后不改变排名。
            for image in data["images"]:
                image["width"] *= 2
                image["height"] *= 2
            for anno in data["annotations"]:
                anno["bbox"] = [2 * value for value in anno["bbox"]]
            for item in predictions.values():
                for prediction in item["predicted_bboxs"]:
                    prediction["bbox"] = [2 * value for value in prediction["bbox"]]
            anno_path.write_text(json.dumps(data))
            pred_path.write_text(json.dumps(predictions))
            self.assertEqual(others.build_ranking(anno_path, pred_path, "loss", mix_images=False), expected)
            # 非法概率不能悄悄进入排序。
            choices[-1]["probs"] = [float("nan"), 0.5]
            pred_path.write_text(json.dumps(predictions))
            with self.assertRaises(ValueError):
                others.build_ranking(anno_path, pred_path, "entropy")

    def test_joblib_output_is_plain_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            expected = [2, 0, "a.jpg"]
            for mixed in (False, True):
                with self.subTest(mixed=mixed), patch.object(others, "OUTPUT_ROOT", Path(temp)), patch.object(
                    others, "ENABLE_IMAGE_MIXING", mixed
                ), patch.object(others, "build_ranking", return_value=expected) as build:
                    others.main()
                    self.assertEqual(build.call_count, 4)
                    self.assertEqual([call.args[2] for call in build.call_args_list], list(others.BASELINES))
                    for baseline in others.BASELINES:
                        output = Path(temp) / (
                            f"{baseline}/{others.DATASET_NAME}/{others.MODEL_NAME}/"
                            f"{others.FAULT_RATIO}/repeat_{others.REPEAT_ID}/rank.joblib"
                        )
                        self.assertEqual(joblib.load(output), expected)

    def test_percentile_ties_and_empty_groups(self):
        result = others.within_group_percentiles([(4, 1), (2, 5), (3, 3), (1, 5)], descending=True)
        self.assertEqual(result, [(1, 0.25), (2, 0.25), (3, 0.625), (4, 0.875)])
        self.assertEqual(others.within_group_percentiles([]), [])
        self.assertEqual(others.merge_rankings([9], [(2, 5), (1, 10)],
                                             [("b.jpg", 0.2), ("a.jpg", 0.1), ("c.jpg", 1)]),
                         [9, 1, "a.jpg", 2, "b.jpg", "c.jpg"])
        self.assertEqual(others.merge_rankings([], [], [("b.jpg", 1), ("a.jpg", 0.2)]),
                         ["a.jpg", "b.jpg"])
        self.assertEqual(others.merge_rankings([9], [(1, 5)], []), [9, 1])
        self.assertEqual(others.merge_rankings([], [(1, 5)], [("a.jpg", 1)]), [1, "a.jpg"])
        with patch.object(others, "IMAGE_PERCENTILE_SCALE", 0.1):
            self.assertEqual(others.merge_rankings([], [(1, 5)], [("a.jpg", 0.2)]), ["a.jpg", 1])

    def test_image_scores_reuse_objectlab_dataset_minimum(self):
        from baselines_new import objectLab
        by_image = {"a.jpg": [("0", 0, [0, 0, 10, 10])]}
        image_info = {name: {"width": 100, "height": 100} for name in ("a.jpg", "b.jpg", "c.jpg")}
        far = {"bbox": [80, 80, 90, 90], "conf": 0.99, "predicted_cls": 0}
        absent = {"bbox": [0, 0, 10, 10], "conf": 0.99, "predicted_cls": 1}
        predictions = {"a.jpg": {"predicted_bboxs": [far]}, "b.jpg": {"predicted_bboxs": [absent]}}
        _, sim = objectLab.box_matrices(by_image["a.jpg"], [far], (100, 100))
        scores = dict(others.collect_image_scores(by_image, image_info, predictions))
        self.assertAlmostEqual(scores["a.jpg"], float(sim[0, 0]))
        self.assertAlmostEqual(scores["b.jpg"], float(sim[0, 0]) * 0.01)
        self.assertEqual(scores["c.jpg"], 1)


if __name__ == "__main__":
    unittest.main()
