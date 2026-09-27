import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from tunequest.common import normalize_code, make_prompt, binary_metrics, best_threshold
from tunequest.prediction import submission
from tunequest.cli import main, read_training_config


class CoreTests(unittest.TestCase):
    def test_normalization_preserves_internal_indent(self):
        self.assertEqual(normalize_code("\r\nx\r\n    y  \r\n"), "x\n    y")

    def test_prompt_regression(self):
        self.assertEqual(make_prompt("a", "b"), "Determine if Function 1 and Function 2 are semantically equivalent.\n### Function 1:\na\n### Function 2:\nb\n### Equivalent:")

    def test_metrics_and_threshold_ties(self):
        result = binary_metrics([0, 1, 1], [0.1, 0.5, 0.2], 0.5)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 2/3)
        self.assertEqual(best_threshold([0, 1], [0, 1])["threshold"], 0.05)
        self.assertEqual(best_threshold([0, 0], [0, 0])["threshold"], 0.5)

    def test_reject_invalid_metrics(self):
        for y, p, t in [([], [], .5), ([1], [], .5), ([2], [.4], .5), ([1], [float('nan')], .5), ([1], [.4], 2)]:
            with self.assertRaises(ValueError):
                binary_metrics(y, p, t)

    def test_submission_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'p.csv', Path(tmp)/'out.csv'
            source.write_text('id,p_final\na,0.37\nb,0.36\n')
            submission(source, target, .37)
            with target.open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [{'id': 'a', 'label': '1'}, {'id': 'b', 'label': '0'}])
            with self.assertRaises(FileExistsError):
                submission(source, target, .37)

    def test_submission_rejects_duplicates_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'p.csv', Path(tmp)/'out.csv'
            source.write_text('id,p_final\na,0.2\na,0.3\n')
            with self.assertRaises(ValueError):
                submission(source, target, .5)
            self.assertFalse(target.exists())

    def test_config_and_cli(self):
        config = read_training_config('configs/bf16.json')
        self.assertEqual(config['warmup_steps'], 470)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            main(['train', '--check-config'])
        self.assertEqual(json.loads(stream.getvalue()), config)


if __name__ == '__main__':
    unittest.main()
