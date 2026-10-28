"""CSV/plot orchestration contracts: no estimator fitting or solver calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR

SOURCE = Path(os.environ.get("FIBERNET_ML_UTILS_SOURCE", Path(__file__).parents[1] / "fibernet/ml/utils.py"))
spec = importlib.util.spec_from_file_location("csv_workflow_utils", SOURCE)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class WorkflowContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for cls in (RandomForestRegressor, Ridge, SVR, MLPRegressor, StandardScaler, Pipeline):
            p = patch.object(cls, "fit", side_effect=AssertionError("fitting forbidden"))
            p.start()
            self.addCleanup(p.stop)

    def csv(self, **columns):
        path = self.root / "input.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        return path

    def reject(self, path, message, **kwargs):
        output = self.root / "never-created"
        with patch.object(utils, "train_predictor", side_effect=AssertionError("training reached")), patch.object(utils, "cross_validate", side_effect=AssertionError("CV reached")):
            with self.assertRaisesRegex(ValueError, message):
                utils.predict_from_csv(path, output_dir=str(output), **kwargs)
        self.assertFalse(output.exists())

    def test_batch_csv_directs_existing_bridge(self):
        self.reject(self.csv(grid_x=range(10), grid_y=range(10), max_force=range(10)), "train_model")

    def test_missing_target(self):
        self.reject(self.csv(feat_x=range(10), label_force=range(10)), "target")

    def test_target_overlap_before_variance_filter(self):
        self.reject(self.csv(feat_x=range(10), feat_y=[1]*10), "target", target="feat_y")

    def test_label_prefix_never_admits_labels(self):
        self.reject(self.csv(label_other=range(10), max_force=range(10)), "label_", feature_prefix="label_")

    def test_empty_prefix_rejects_target(self):
        self.reject(self.csv(x=range(10), max_force=range(10)), "target", feature_prefix="")

    def test_constant_features_rejected(self):
        self.reject(self.csv(feat_x=[1]*10, max_force=range(10)), "feature")

    def ridge_pipeline(self):
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
        model[-1].coef_ = np.array([-2., 3.])
        return model

    def test_ridge_final_estimator_coefficients(self):
        model = self.ridge_pipeline()
        model[0].coef_ = np.array([999., 999.])
        fig = utils.plot_feature_importance(model, ["a", "b"], top_k=2)
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(sorted(p.get_width() for p in fig.axes[0].patches), [2., 3.])
        self.assertIn("coefficient", fig.axes[0].get_xlabel().lower())

    def test_direct_coefficients(self):
        model = Ridge(); model.coef_ = np.array([-2., 3.])
        fig = utils.plot_feature_importance(model, ["a", "b"])
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(sorted(p.get_width() for p in fig.axes[0].patches), [2., 3.])

    def test_unscaled_pipeline_label(self):
        model = self.ridge_pipeline()
        model.steps[0] = ("scaler", StandardScaler(with_mean=False, with_std=False))
        fig = utils.plot_feature_importance(model, ["a", "b"])
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(fig.axes[0].get_xlabel(), "Absolute coefficient (model feature space)")

    def test_feature_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "feature"):
            utils.plot_feature_importance(self.ridge_pipeline(), ["a"])

    def test_multioutput_refused(self):
        model = Ridge(); model.coef_ = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            utils.plot_feature_importance(model, ["a", "b", "c", "d"])

    def test_nonfinite_importance_refused(self):
        model = Ridge(); model.coef_ = np.array([np.nan, 1.])
        with self.assertRaisesRegex(ValueError, "finite"):
            utils.plot_feature_importance(model, ["a", "b"])

    def test_complex_importance_refused(self):
        model = Ridge(); model.coef_ = np.array([1+2j, 3+4j])
        with self.assertRaisesRegex(ValueError, "real-valued"):
            utils.plot_feature_importance(model, ["a", "b"])

    def test_direct_estimator_unrelated_steps(self):
        model = Ridge(); model.coef_ = np.array([2., 3.])
        model.steps = 10
        fig = utils.plot_feature_importance(model, ["a", "b"])
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(sorted(p.get_width() for p in fig.axes[0].patches), [2., 3.])

    def test_direct_tree_importance_unchanged(self):
        model = RandomForestRegressor()
        with patch.object(RandomForestRegressor, "feature_importances_", new_callable=PropertyMock, return_value=np.array([.2, .8])):
            fig = utils.plot_feature_importance(model, ["a", "b"])
        self.addCleanup(__import__("matplotlib.pyplot", fromlist=["close"]).close, fig)
        self.assertEqual(sorted(p.get_width() for p in fig.axes[0].patches), [.2, .8])

    def test_actual_plot_save_error_propagates(self):
        destination = self.root / "not-directory"
        destination.write_text("preserve")
        with self.assertRaises(OSError):
            utils.plot_feature_importance(self.ridge_pipeline(), ["a", "b"], save_path=str(destination/"plot.png"))
        self.assertEqual(destination.read_text(), "preserve")
        __import__("matplotlib.pyplot", fromlist=["close"]).close("all")

    def test_transformed_pipeline_refused(self):
        model = Pipeline([("poly", PolynomialFeatures()), ("ridge", Ridge())])
        model[-1].coef_ = np.ones(2)
        with self.assertRaisesRegex(ValueError, "StandardScaler|feature-preserving"):
            utils.plot_feature_importance(model, ["a", "b"])

    def test_direct_unsupported_refused(self):
        with self.assertRaises(ValueError):
            utils.plot_feature_importance(SVR(), ["a", "b"])

    def workflow(self, model, output=True, importance_error=None):
        path = self.csv(feat_b=range(20), feat_a=np.arange(20)*2, max_force=range(20))
        out = self.root / "output"
        metrics = {"r2": .5, "rmse": .2}
        cv = {"mean_r2": .4, "std_r2": .1}
        def plot(*args, **kwargs):
            Path(kwargs["save_path"]).write_bytes(b"plot-test-double")
        def importance(*args, **kwargs):
            if importance_error: raise importance_error
            plot(*args, **kwargs)
        with patch.object(utils, "train_predictor", return_value=(model, metrics)) as train, patch.object(utils, "cross_validate", return_value=cv), patch.object(type(model), "predict", return_value=np.zeros(4)), patch("pickle.dump", side_effect=lambda obj, stream: stream.write(b"serialization-test-double")), patch.object(utils, "plot_predictions", side_effect=plot), patch.object(utils, "plot_residuals", side_effect=plot), patch.object(utils, "plot_feature_importance", side_effect=importance) as imp:
            result = utils.predict_from_csv(path, output_dir=str(out) if output else None)
        self.assertEqual(train.call_args.kwargs["feature_names"], ["feat_b", "feat_a"])
        self.assertEqual(train.call_args.args[0].shape, (20, 2))
        return result, out, imp

    def test_no_output_status(self):
        result, out, imp = self.workflow(self.ridge_pipeline(), output=False)
        self.assertEqual(result["importance_plot"]["status"], "not_requested")
        self.assertFalse(out.exists()); imp.assert_not_called()

    def test_ridge_written_status(self):
        result, out, imp = self.workflow(self.ridge_pipeline())
        self.assertEqual(result["importance_plot"]["status"], "written")
        self.assertEqual(json.loads((out/"metrics.json").read_text())["importance_plot"]["status"], "written")
        self.assertTrue((out/"importance.png").exists()); imp.assert_called_once()
        self.assertTrue({"model", "metrics", "cv_results", "feature_names", "target", "n_samples"} <= result.keys())

    def test_unsupported_skip_preserves_old_file(self):
        for model in (SVR(), MLPRegressor(), utils._build_model("svm"), utils._build_model("mlp")):
            with self.subTest(model=type(model).__name__):
                out = self.root/"output"; out.mkdir(exist_ok=True)
                (out/"importance.png").write_bytes(b"previous-image")
                result, out, imp = self.workflow(model)
                self.assertEqual(result["importance_plot"]["status"], "skipped")
                self.assertIn("not updated", result["importance_plot"]["reason"])
                self.assertEqual((out/"importance.png").read_bytes(), b"previous-image")
                self.assertEqual(json.loads((out/"metrics.json").read_text())["importance_plot"], result["importance_plot"])
                for name in ["metrics.json", "model.pkl", "predictions.png", "residuals.png"]: self.assertTrue((out/name).exists())
                imp.assert_not_called()

    def test_malformed_importance_fails_before_output(self):
        model = self.ridge_pipeline(); model[-1].coef_ = np.ones((2, 2))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.workflow(model)
        self.assertFalse((self.root/"output").exists())

    def test_genuine_plot_errors_propagate(self):
        for error in (ValueError("plot-internal"), OSError("disk-error")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaisesRegex(type(error), str(error)):
                    self.workflow(self.ridge_pipeline(), importance_error=error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
