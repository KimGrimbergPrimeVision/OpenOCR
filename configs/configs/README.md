# Custom SVTRv2 training

Run all commands from the OpenOCR repository root. These configs use the
repository's native `tools/train_rec.py` pipeline, not a separate training framework.

## Configuration

- `training.yaml` inherits the SVTRv2 Tiny RCTC architecture and uses
  `SimpleDataSet` for cropped text images and tab-separated labels.
- `preprocessing.yaml` supplies the shared image size, padding and CSV split settings.
- `normalization.yaml` controls lowercase conversion, diacritic replacement and
  space handling. Whitespace is collapsed to single spaces; surrounding whitespace
  is removed. Unlisted diacritics are decomposed and combining marks removed.
- `evaluation.yaml` evaluates the held-out test split with `best.pth`.
- `inference.yaml` uses the same architecture, dictionary and image preprocessing.
- `datasets.yaml` is optional metadata only; OpenOCR does not load it automatically.

`dictionaries/characters.txt` contains one UTF-8 character per line. Do not add
`<space>` or a CTC blank token: OpenOCR adds the space via `use_space_char` and
the CTC blank itself. The decoder's class count is derived from this dictionary.
Unknown characters are rejected by default, not silently deleted. The punctuation
listed in `normalization.yaml` must be present in the dictionary.

The label limit is **125 normalized characters**, including spaces. Images are
resized proportionally and padded to **32 x 1024**. With this architecture that
produces 256 CTC steps, covering even 125 repeated characters (249 steps required).
Longer inputs use more GPU memory; the initial batch size is 4. Metrics preserve
spaces and punctuation rather than applying the benchmark's alphanumeric filtering.

## Prepare labels

The source CSV is `Data/mazda_all_clean/labels.csv`, with columns `crop_path`,
`source_id` and `label`. Image paths are relative to the configured dataset root.
The `source_id` is a grouping key, not an image path that needs to exist locally.

First validate without writing:

```powershell
python tools/prepare_rec_labels.py --config ./configs/configs/training.yaml --dry-run
```

Fix invalid labels before preparing data. In particular, the current CSV contains
`{` markers that are not in the dictionary. Decide whether those markers are
annotation syntax to remove during data cleaning or actual characters to retain
in the dictionary. Do not remove them automatically without confirming their meaning.
To deliberately exclude invalid samples, pass `--skip-invalid`.

Then create `train.txt`, `val.txt` and `test.txt`:

```powershell
python tools/prepare_rec_labels.py --config ./configs/configs/training.yaml
```

Empty labels are excluded. Remaining samples are split approximately 80/10/10 by
source document, preventing lines from the same document appearing in different
splits. The seed is fixed; existing split files are never overwritten automatically.
Prepared labels use `image_path<TAB>normalized_text`, not CSV or JSON.

## Train, Evaluate And Recognize

```powershell
python tools/train_rec.py --config ./configs/configs/training.yaml
python tools/eval_rec.py --config ./configs/configs/evaluation.yaml
python tools/infer_rec.py --config ./configs/configs/inference.yaml
```

Training starts from scratch unless a compatible pretrained checkpoint is supplied.
Checkpoints using a different architecture or dictionary are not interchangeable.
Resume interrupted training with:

```powershell
python tools/train_rec.py --config ./configs/configs/training.yaml --opt Global.checkpoints=./output/rec/mazda/svtrv2_tiny_rctc/latest.pth
```

Focused integration tests, with no training or real dataset writes:

```powershell
python -m unittest discover -s tests -p test_custom_rec_training.py -v
```