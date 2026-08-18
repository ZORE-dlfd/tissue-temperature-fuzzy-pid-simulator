import csv
import json
import tempfile
import unittest
from pathlib import Path

from simulation_core import SimulationConfig, export_data_files, run_simulation


class ExportTests(unittest.TestCase):
    def test_matplotlib_can_write_a_nonempty_png(self):
        from matplotlib.figure import Figure

        result = run_simulation(SimulationConfig(duration=2.0, dt=0.1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.png"
            figure = Figure(figsize=(4, 3))
            axis = figure.subplots()
            axis.plot(result.time, result.true_temp)
            figure.savefig(path, dpi=100)
            self.assertGreater(path.stat().st_size, 1_000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_export_contains_required_data_and_provenance_warning(self):
        result = run_simulation(SimulationConfig(duration=2.0, dt=0.1))
        with tempfile.TemporaryDirectory() as directory:
            paths = export_data_files(result, Path(directory))
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["readme"].exists())
            with paths["csv"].open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(
                header,
                [
                    "time_s",
                    "target_temp_c",
                    "true_temp_c",
                    "measurement_temp_c",
                    "estimated_temp_c",
                    "feedback_temp_c",
                    "error_c",
                    "power_percent",
                    "kp",
                    "ki",
                    "kd",
                ],
            )
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertIn("metrics", payload)
            warning = paths["readme"].read_text(encoding="utf-8")
            self.assertIn("不是论文原始实验参数", warning)

    def test_export_refuses_existing_directory(self):
        result = run_simulation(SimulationConfig(duration=2.0, dt=0.1))
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing"
            existing.mkdir()
            (existing / "timeseries.csv").touch()
            with self.assertRaises(FileExistsError):
                export_data_files(result, existing)


if __name__ == "__main__":
    unittest.main()
