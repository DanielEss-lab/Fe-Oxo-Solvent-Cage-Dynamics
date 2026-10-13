"""Regression and workflow checks using synthetic coordinates only."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from bgan_eps_feoxo.analysis import summarize
from bgan_eps_feoxo.bgan import BGANEPS
from bgan_eps_feoxo.main import load_reaction_coordinate, main


def inputs():
    rng = np.random.RandomState(42)
    n = 256
    return np.column_stack((rng.uniform(1.31, 3.89, n),
                            rng.uniform(1.2, 1.6, (n, 2)),
                            rng.uniform(0.9, 2.2, (n, 2)),
                            rng.uniform(-1.5, 1.5, (n, 2))))


TOPOLOGY = "5\n1 2\n2 3\n1 2 3\n2 3 4\n1 2 3 4\n2 3 4 5\n"


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "log" / "synthetic"
        self.run.mkdir(parents=True)
        (self.root / "dataset").mkdir()
        np.save(self.run / "dof.npy", inputs())
        (self.run / "topology.txt").write_text(TOPOLOGY)
        self.options = ["--workdir", str(self.root), "--reaction", "synthetic",
                        "--rcmin", "1.3", "--rcmax", "3.9", "--ensemble", "4",
                        "--window_points", "4", "--bins", "5", "--batch_size", "32",
                        "--epochs", "2", "--hidden_nodes", "16", "--latent_dim", "8",
                        "--loop", "2", "--seed", "42"]

    def test_eps_regression_both_directions(self):
        # Numerical reference obtained from the supplied unmodified working
        # implementation before packaging, not recomputed by these tests.
        expected = np.array([-0.2834710566244625, -0.031585055212343234,
                             -0.30819954498074775, -0.442102266544967])
        for direction in ("decreasing", "increasing"):
            obj = BGANEPS(str(self.run))
            obj.dofs = inputs().T
            obj.real_mask = np.ones(256, dtype=bool)
            args = SimpleNamespace(eps_type="average", bins=5, temperature=298.15,
                                   ensemble=4, window_points=4, rc_direction=direction,
                                   min_window_samples=1, bondmin=1.3, bondmax=3.9)
            obj.EPS(args, 0)
            actual = np.load(self.run / "bgan1" / "eps.npy")
            np.testing.assert_allclose(actual, expected if direction == "decreasing"
                                       else expected[::-1], atol=1e-12, rtol=1e-12)

    def test_training_custom_path_summary_and_overwrite_guard(self):
        torch.set_num_threads(1)
        rc = inputs()[:, 0]
        np.save(self.root / "dataset" / "rc.npy", rc)
        before = (self.run / "dof.npy").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            main(self.options + ["--rc_file", "dataset/rc.npy"])
        ep = np.load(self.run / "synthetic_eps.npy")
        self.assertEqual(ep.shape, (2, 4))
        self.assertTrue(np.isfinite(ep).all())
        self.assertEqual(before, (self.run / "dof.npy").read_bytes())
        settings = json.loads((self.run / "run_settings.json").read_text())
        self.assertEqual(settings["loop_seeds"], [42, 43])
        mean = summarize(self.run, 2, self.root / "summary")
        np.testing.assert_allclose(mean, (ep - ep[:, [1]]).mean(axis=0))
        self.assertEqual(mean[1], 0)
        with self.assertRaises(FileExistsError):
            main(self.options)

    def test_coordinate_length_mismatch(self):
        np.save(self.root / "short.npy", np.zeros(3))
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            load_reaction_coordinate(str(self.root / "short.npy"), 256)

    def test_nonfinite_input_rejected_before_training(self):
        data = inputs()
        data[0, 1] = np.nan
        np.save(self.run / "dof.npy", data)
        with self.assertRaisesRegex(ValueError, "NaN"):
            main(self.options)
        self.assertFalse((self.run / "bgan1").exists())

    def test_topology_mismatch_rejected_before_training(self):
        (self.run / "topology.txt").write_text("5\n1 2\n")
        with self.assertRaisesRegex(ValueError, "topology entry"):
            main(self.options)

    def test_invalid_reference_window(self):
        folder = self.run / "bgan1"
        folder.mkdir()
        np.save(folder / "eps.npy", [0., 1.])
        (folder / "eps.log").write_text("header\n1 3 2 0 0 100 50\n2 2 1 1 1 100 50\n")
        with self.assertRaisesRegex(ValueError, "Reference window"):
            summarize(self.run, 3, self.root / "summary")

    def test_single_loop_uncertainty_is_undefined(self):
        folder = self.run / "bgan1"
        folder.mkdir()
        np.save(folder / "eps.npy", [0., 1.])
        (folder / "eps.log").write_text("header\n1 3 2 0 0 100 50\n2 2 1 1 1 100 50\n")
        summarize(self.run, 1, self.root / "summary")
        with (self.root / "summary" / "entropy_profile.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(np.isnan(float(rows[0]["sem"])))

    def test_empty_window_not_silently_summarized(self):
        folder = self.run / "bgan1"
        folder.mkdir()
        np.save(folder / "eps.npy", [np.nan, 1.])
        (folder / "eps.log").write_text("header\n1 3 2 nan nan 0 0\n2 2 1 1 nan 100 50\n")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            summarize(self.run, 2, self.root / "summary")


if __name__ == "__main__":
    unittest.main()
