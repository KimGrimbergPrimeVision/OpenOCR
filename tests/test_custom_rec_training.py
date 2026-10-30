import copy
import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from openrec.preprocess import create_operators
from tools.engine.config import Config
from tools.prepare_rec_labels import prepare_labels


class CustomRecTrainingTests(unittest.TestCase):

    def setUp(self):
        self.config = Config('configs/configs/training.yaml').cfg
        self.normalize, self.encode = create_operators(
            self.config['Train']['dataset']['transforms'][1:3], self.config['Global'])

    def test_normalization(self):
        self.assertEqual(self.normalize({'label': '  HÉLLO   Straße '})['label'],
                         'hello strasse')
        self.assertEqual(self.normalize({'label': 'E\u0301cole, 12/3.'})['label'],
                         'ecole, 12/3.')
        with self.assertRaisesRegex(ValueError, 'unknown characters'):
            self.normalize({'label': 'house {13{'})
        with self.assertRaises(ValueError):
            self.normalize({'label': ' '})

    def test_length_limit(self):
        result = self.encode(self.normalize({'label': 'a' * 125}))
        self.assertEqual(int(result['length']), 125)
        self.assertEqual(len(result['label']), 125)
        with self.assertRaises(ValueError):
            self.normalize({'label': 'a' * 126})

    def test_resize_and_inference_filter(self):
        from tools.infer_rec import build_rec_process

        for name in ['training', 'evaluation', 'inference']:
            config = Config(f'configs/configs/{name}.yaml').cfg
            self.assertEqual(config['Global']['max_text_length'], 125)
            self.assertEqual(config['Global']['image_shape'], [32, 1024])
        operators, ratio_resize = build_rec_process(copy.deepcopy(self.config))
        self.assertFalse(ratio_resize)
        self.assertFalse(any('Label' in next(iter(operator)) for operator in operators))
        resize = create_operators([{'RecTVResize': None}], self.config['Global'])[0]
        self.assertEqual(tuple(resize({'image': Image.new('RGB', (250, 32))})['image'].shape),
                         (3, 32, 1024))

    def test_ctc_capacity(self):
        from openrec.modeling import build_model
        from openrec.postprocess import build_post_process

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            postprocess = build_post_process(self.config['PostProcess'], self.config['Global'])
            architecture = copy.deepcopy(self.config['Architecture'])
            architecture['Decoder']['out_channels'] = postprocess.get_character_num()
            model = build_model(architecture).eval()
            with torch.inference_mode():
                predictions = model(torch.zeros(1, 3, 32, 1024))
            self.assertEqual(predictions.shape[1], 256)
            self.assertEqual(predictions.shape[2], len(self.encode.character))
            self.assertGreaterEqual(predictions.shape[1], 2 * 125 - 1)
            logits = torch.randn(predictions.shape[1], 1, predictions.shape[2])
            loss = torch.nn.CTCLoss(zero_infinity=False)(
                logits.log_softmax(2), torch.ones(1, 125, dtype=torch.long),
                torch.tensor([predictions.shape[1]]), torch.tensor([125]))
            self.assertTrue(torch.isfinite(loss).item())
        finally:
            torch.set_num_threads(previous_threads)

    def test_csv_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = copy.deepcopy(self.config)
            config['Train']['dataset']['data_dir'] = str(root)
            config['DataPreparation']['input_csv'] = str(root / 'labels.csv')
            config['DataPreparation']['output_dir'] = str(root)
            with (root / 'labels.csv').open('w', encoding='utf-8', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['crop_path', 'source_id', 'label'])
                writer.writeheader()
                for source in range(10):
                    for line in range(2):
                        name = f'{source}_{line}.png'
                        Image.new('RGB', (50, 32)).save(root / name)
                        writer.writerow({'crop_path': name, 'source_id': str(source),
                                         'label': 'HÉLLO, Straße'})
                writer.writerow({'crop_path': 'empty.png', 'source_id': '0', 'label': ''})
                Image.new('RGB', (50, 32)).save(root / 'invalid.png')
                writer.writerow({'crop_path': 'invalid.png', 'source_id': '0',
                                 'label': 'house {13{'})
            with self.assertRaisesRegex(ValueError, '1 invalid samples'):
                prepare_labels(config)
            self.assertFalse((root / 'train.txt').exists())
            counts = prepare_labels(config, dry_run=True, skip_invalid=True)
            self.assertFalse((root / 'train.txt').exists())
            self.assertEqual(sum(counts[split] for split in ['train', 'val', 'test']), 20)
            self.assertEqual(counts['empty'], 1)
            self.assertEqual(counts['invalid'], 1)
            self.assertEqual(prepare_labels(config, skip_invalid=True), counts)
            sources = []
            for split in ['train', 'val', 'test']:
                lines = (root / f'{split}.txt').read_text(encoding='utf-8').splitlines()
                self.assertTrue(all(line.endswith('\thello, strasse') for line in lines))
                sources.append({line.split('_')[0] for line in lines})
            self.assertFalse(sources[0] & sources[1] or sources[0] & sources[2]
                             or sources[1] & sources[2])
            with self.assertRaises(FileExistsError):
                prepare_labels(config, skip_invalid=True)


class RecognitionMetricTests(unittest.TestCase):

    def test_corpus_cer_across_batches(self):
        from openrec.metrics.rec_metric import RecMetric

        metric = RecMetric(main_indicator='cer', is_lower=False, ignore_space=False)
        self.assertFalse(metric.greater_is_better)
        metric(([('x', 1)], [('a', 1)]))
        metric(([('abcdefghij', 1)], [('abcdefghij', 1)]))
        self.assertAlmostEqual(metric.get_metric()['cer'], 1 / 11)
        self.assertEqual(metric.get_metric()['cer'], 0)

    def test_cer_insertions_spaces_and_punctuation(self):
        from openrec.metrics.rec_metric import RecMetric

        metric = RecMetric(is_lower=False, ignore_space=False)
        metric(([('abcd', 1)], [('a', 1)]))
        self.assertEqual(metric.get_metric()['cer'], 3)
        metric(([('ab', 1)], [('a b!', 1)]))
        self.assertEqual(metric.get_metric()['cer'], 0.5)
        self.assertTrue(metric.greater_is_better)

    def test_ratio_metric_cer(self):
        from openrec.metrics.rec_metric import RecMetric

        metric = RecMetric(with_ratio=True, is_lower=False)
        metric(([('cat', 1)], [('car', 1)]), batch=[[1]])
        self.assertAlmostEqual(metric.get_metric()['cer'], 1 / 3)


if __name__ == '__main__':
    unittest.main()