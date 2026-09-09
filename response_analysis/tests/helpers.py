from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from response_analysis.examples import make_run


class RunTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="response-analysis-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make(self, name="run", **kwargs):
        kwargs.setdefault("channels", ["angle.roll"])
        return make_run(self.root / name, **kwargs)

    def bare(self, name="flight", **kwargs):
        source = self.make(f"{name}-source", **kwargs)
        csv_path = self.root / f"{name}.csv"
        csv_path.write_bytes((source / "samples.csv").read_bytes())
        return csv_path

    def samples(self, path, edit):
        filename = path / "samples.csv"
        frame = pd.read_csv(filename)
        changed = edit(frame)
        with np.errstate(invalid="ignore"):
            (frame if changed is None else changed).to_csv(filename, index=False)

    def manifest(self, **groups):
        return {
            "reference_group": "real", "baseline_group": "before",
            "groups": {name: [str(path) for path in paths] for name, paths in groups.items()},
        }
