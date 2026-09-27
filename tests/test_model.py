"""Optional CPU smoke test with random tiny Llama weights, never a benchmark."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

available = all(importlib.util.find_spec(x) for x in ('torch', 'transformers', 'peft', 'tokenizers'))


@unittest.skipUnless(available, 'Optional ML dependencies absent')
class AdapterTests(unittest.TestCase):
    def test_local_adapter_loading_and_tta(self):
        import torch
        from transformers import LlamaConfig, LlamaForSequenceClassification, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from peft import LoraConfig, get_peft_model
        from tunequest.prediction import predict
        from tunequest.common import binary_metrics
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            base, adapter = Path(tmp)/'base', Path(tmp)/'adapter'
            tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'return': 3, '1': 4, '2': 5}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
            tokenizer.save_pretrained(str(adapter))
            config = LlamaConfig(vocab_size=6, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, num_labels=2, pad_token_id=1)
            model = LlamaForSequenceClassification(config)
            model.save_pretrained(str(base))
            model.config._name_or_path = str(base)
            model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type='SEQ_CLS', modules_to_save=['score']))
            model.peft_config['default'].base_model_name_or_path = str(base)
            model.save_pretrained(str(adapter))
            source, target = Path(tmp)/'input.jsonl', Path(tmp)/'out.csv'
            records = [{'id':'a','func1':'return 1','func2':'return 2'}, {'id':'b','func1':'return 2','func2':'return 1'}]
            source.write_text('\n'.join(json.dumps(r) for r in records))
            result = predict(source, adapter, target, max_length=64, batch_size=2, tta=True, device='cpu')
            self.assertEqual(result['rows'], 2)
            with target.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[0]['p_final']), float(rows[1]['p_final']), places=6)
            for row in rows:
                self.assertAlmostEqual(float(row['p_final']), (float(row['p_forward'])+float(row['p_reverse']))/2)
            binary_metrics([0,1], [float(r['p_final']) for r in rows], .5)


if __name__ == '__main__':
    unittest.main()
