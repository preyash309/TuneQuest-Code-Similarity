import importlib.util
from pathlib import Path
import tempfile
import unittest

from tunequest.preprocessing import prepare_frames, preprocess


@unittest.skipUnless(importlib.util.find_spec('pandas') and importlib.util.find_spec('sklearn'), 'Optional data dependencies absent')
class PreprocessingTests(unittest.TestCase):
    def frame(self):
        import pandas as pd
        return pd.DataFrame([{'id': i, 'func1': f'a{i}', 'func2': f'b{i}', 'label': i%2} for i in range(60)])

    def test_dedup_conflicts_split_and_determinism(self):
        import pandas as pd
        base = self.frame()
        duplicate = {'id': 60, 'func1': 'b0', 'func2': 'a0', 'label': 0}
        conflict = {'id': 61, 'func1': 'b1', 'func2': 'a1', 'label': 0}
        frame = pd.concat([base, pd.DataFrame([duplicate, conflict])], ignore_index=True)
        train, val, report = prepare_frames(frame, sample_size=30, validation_fraction=.2)
        self.assertEqual(report['clean_unique_pairs'], 59)
        self.assertEqual(report['conflicting_unordered_pairs'], 1)
        self.assertEqual(len(train), 30)
        self.assertFalse(set(train.id) & set(val.id))
        keys = lambda df: {tuple(sorted((a,b))) for a,b in zip(df.func1,df.func2)}
        self.assertFalse(keys(train) & keys(val))
        self.assertNotIn(1, set(train.id) | set(val.id))
        again, _, _ = prepare_frames(frame, sample_size=30, validation_fraction=.2)
        self.assertEqual(train.id.tolist(), again.id.tolist())

    def test_full_pool_fallback_and_symmetry(self):
        train, val, report = prepare_frames(self.frame(), sample_size=1000, validation_fraction=.2, symmetry=True)
        self.assertEqual(len(train), 96)
        self.assertEqual(len(val), 12)
        self.assertEqual(set(train.orientation), {'original', 'reversed'})
        self.assertEqual(report['train_validation_pair_overlap'], 0)

    def test_invalid_data(self):
        for change in ('duplicate', 'fractional', 'missing'):
            df = self.frame()
            if change == 'duplicate': df.loc[1, 'id'] = 0
            elif change == 'fractional': df['label'] = df.label.astype(float); df.loc[0, 'label'] = .4
            else: df = df.drop(columns='func1')
            with self.assertRaises(ValueError):
                prepare_frames(df)

    @unittest.skipUnless(importlib.util.find_spec('datasets'), 'datasets absent')
    def test_disk_roundtrip(self):
        from datasets import load_from_disk
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'input.jsonl', Path(tmp)/'dataset'
            self.frame().to_json(source, orient='records', lines=True)
            report = preprocess(source, target, validation_fraction=.2)
            saved = load_from_disk(str(target))
            self.assertEqual(len(saved['train']), report['training_rows'])
            self.assertEqual(len(saved['validation']), 12)
            with self.assertRaises(FileExistsError):
                preprocess(source, target)


if __name__ == '__main__':
    unittest.main()
