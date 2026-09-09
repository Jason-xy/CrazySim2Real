from contextlib import redirect_stderr, redirect_stdout
import csv
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import subprocess
import sys

from response_analysis import analyze_run, load_run, validate_run
from response_analysis.__main__ import main
from response_analysis.examples import generate_demo
from response_analysis.report import write_report
from .helpers import RunTestCase


class ResourceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href"):
                self.resources.append(value)


class CliTests(RunTestCase):
    def invoke(self, *args):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as error:
            status = main(list(map(str, args)))
        return status, output.getvalue(), error.getvalue()

    def check_report(self, output):
        for name in ("report.html", "results.json", "summary.csv"):
            self.assertTrue((output / name).is_file())
        page = (output / "report.html").read_text()
        parsed = ResourceParser()
        parsed.feed(page)
        for resource in parsed.resources:
            self.assertNotIn("://", resource)
            self.assertTrue((output / resource).is_file(), resource)
        result = json.loads((output / "results.json").read_text())
        self.assertIn("software", result)
        self.assertEqual(result["schema_version"], 2)
        for text in ("metadata_sources", '"metadata"', "condition_id", "unknown_"):
            self.assertNotIn(text, json.dumps(result))
        with (output / "summary.csv").open() as stream:
            self.assertNotIn("condition", csv.DictReader(stream).fieldnames)
        return result

    def test_csv_only_validate_analyze_compare_and_preserve_inputs(self):
        reference = self.bare("reference")
        before = self.bare("before", gain=0.7, seed=2)
        original = {path: path.read_bytes() for path in (reference, before)}
        status, stdout, error = self.invoke("validate", reference, "--out", self.root / "validation")
        self.assertEqual(status, 0, error)
        self.assertEqual(json.loads(stdout)["channels"], ["angle.roll"])
        self.check_report(self.root / "validation")
        status, _, error = self.invoke("analyze", reference, "--out", self.root / "analysis")
        self.assertEqual(status, 0, error)
        analysis = self.check_report(self.root / "analysis")
        self.assertEqual(analysis["checksum"], load_run(reference).checksum)
        filename = self.root / "comparison.json"
        filename.write_text(json.dumps(self.manifest(real=[reference], before=[before])))
        status, _, error = self.invoke("compare", filename, "--out", self.root / "comparison")
        self.assertEqual(status, 0, error)
        self.check_report(self.root / "comparison")
        for path, contents in original.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertFalse(path.with_suffix(".metadata.json").exists())

    def test_directory_mode_and_ignored_companion_files(self):
        path = self.make()
        companion = path / "metadata.json"
        companion.write_text("{not valid JSON")
        status, _, error = self.invoke("analyze", path, "--out", self.root / "analysis")
        self.assertEqual(status, 0, error)
        self.check_report(self.root / "analysis")
        self.assertEqual(companion.read_text(), "{not valid JSON")

    def test_output_cannot_overwrite_source_or_existing_files(self):
        directory = self.make()
        direct = self.bare()
        for source in (directory, direct):
            result = analyze_run(source)
            with self.assertRaises(ValueError):
                write_report(result, source)
        with self.assertRaises(ValueError):
            write_report(analyze_run(directory), directory / "nested")
        output = self.root / "output"
        output.mkdir()
        (output / "keep.txt").write_text("untouched")
        with self.assertRaises(ValueError):
            write_report(analyze_run(direct), output)
        self.assertEqual((output / "keep.txt").read_text(), "untouched")

    def test_output_aliases_cannot_overwrite_csv(self):
        path = self.bare()
        alias = self.root / "output-alias"
        alias.symlink_to(path)
        for result in (analyze_run(path), validate_run(path)):
            with self.assertRaisesRegex(ValueError, "source"):
                write_report(result, alias)

    def test_intervals_override_config_and_are_included_in_reports(self):
        path = self.bare()
        config = self.root / "settings.json"
        config.write_text('{"analysis_intervals_s": [[0, 5]]}')
        status, _, error = self.invoke(
            "analyze", path, "--config", config, "--interval", 10, 20,
            "--interval", 25, 35, "--out", self.root / "interval-report",
        )
        self.assertEqual(status, 0, error)
        result = self.check_report(self.root / "interval-report")
        self.assertEqual(result["analysis_intervals_s"], [[10, 20], [25, 35]])
        self.assertEqual(result["config"]["analysis_intervals_s"], [[10, 20], [25, 35]])
        status, _, error = self.invoke(
            "analyze", path, "--config", config, "--out", self.root / "config-report",
        )
        self.assertEqual(status, 0, error)
        self.assertEqual(self.check_report(self.root / "config-report")["analysis_intervals_s"], [[0, 5]])

    def test_invalid_and_insufficient_exit_codes(self):
        self.assertEqual(self.invoke("validate", self.root / "missing.csv")[0], 2)
        path = self.make(duration_s=1)
        status, _, error = self.invoke("analyze", path, "--out", self.root / "report")
        self.assertEqual(status, 3, error)
        self.check_report(self.root / "report")
        status, _, _ = self.invoke("analyze", path, "--interval", 0, 10, "--out", self.root / "unused")
        self.assertEqual(status, 2)
        self.assertFalse((self.root / "unused").exists())
        status, _, _ = self.invoke("analyze", path, "--interval", 2, 1, "--out", self.root / "unused")
        self.assertEqual(status, 2)

    def test_html_escapes_file_labels(self):
        path = self.bare("<b>flight</b>".replace("/", "_"))
        output = self.root / "output"
        write_report(analyze_run(path), output)
        page = (output / "report.html").read_text()
        self.assertNotIn("<b>flight", page)
        self.assertIn("&lt;b&gt;flight", page)

    def test_demo_contains_only_csv_runs_and_comparison_manifest(self):
        manifest = generate_demo(self.root / "demo")
        spec = json.loads(manifest.read_text())
        self.assertEqual(set(spec), {"reference_group", "baseline_group", "groups"})
        for group in spec["groups"].values():
            self.assertEqual(len(group), 3)
            for name in group:
                directory = manifest.parent / name
                self.assertEqual({path.name for path in directory.iterdir()}, {"samples.csv"})

    def test_import_and_module_entrypoint_are_offline(self):
        root = Path(__file__).resolve().parents[2]
        process = subprocess.run(
            [sys.executable, "-c", "import response_analysis,sys; "
             "assert not any(x.startswith(('cflib','tkinter','crazyflie_sim','crazyflie_benchmark')) for x in sys.modules)"],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        process = subprocess.run(
            [sys.executable, "-m", "response_analysis", "analyze", "--help"],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--interval", process.stdout)
        self.assertNotIn("metadata", process.stdout)
