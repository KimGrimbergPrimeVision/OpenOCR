import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openrec.preprocess import create_operators
from tools.engine.config import Config


def prepare_labels(config, dry_run=False, skip_invalid=False):
    settings = config['DataPreparation']
    data_root = Path(config['Train']['dataset']['data_dir']).resolve()
    source = Path(settings['input_csv'])
    output = Path(settings['output_dir'])
    validation_fraction = settings['validation_fraction']
    test_fraction = settings['test_fraction']
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1
            and validation_fraction + test_fraction < 1):
        raise ValueError('Validation/test fractions must be positive and sum to less than 1.')
    normalization = [operator for operator in config['Train']['dataset']['transforms']
                     if 'NormalizeLabel' in operator]
    if len(normalization) != 1:
        raise ValueError('Expected exactly one NormalizeLabel transform.')
    normalize = create_operators(normalization, config['Global'])[0]
    extensions = set(settings['extensions'])
    groups = {}
    errors = []
    seen = set()
    empty_count = 0
    with source.open(encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        required = {'crop_path', 'source_id', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain columns: {sorted(required)}')
        for line_number, row in enumerate(reader, start=2):
            label = row['label']
            if label is not None and not label.strip():
                empty_count += 1
                continue
            try:
                if label is None or not row['source_id']:
                    raise ValueError('Missing label or source_id.')
                relative_path = row['crop_path'].replace('\\', '/')
                if any(character in relative_path for character in '\t\r\n'):
                    raise ValueError('Image path contains a tab or newline.')
                image_path = (data_root / relative_path).resolve()
                if image_path in seen:
                    raise ValueError('Duplicate crop_path.')
                if image_path.suffix.lower() not in extensions or not image_path.is_file():
                    raise ValueError('Image is missing or has an unsupported extension.')
                normalized = normalize({'label': label})['label']
                groups.setdefault(row['source_id'], []).append(
                    f'{relative_path}\t{normalized}\n')
                seen.add(image_path)
            except (ValueError, AttributeError) as error:
                errors.append(f'CSV line {line_number}: {error}')
    if errors and not skip_invalid:
        examples = '\n'.join(errors[:10])
        raise ValueError(f'{len(errors)} invalid samples; no files written.\n{examples}\n'
                         'Fix the labels or explicitly pass --skip-invalid.')
    sources = sorted(groups)
    if len(sources) < 3:
        raise ValueError('At least three source documents with valid labels are required.')
    random.Random(settings['seed']).shuffle(sources)
    validation_count = max(1, int(len(sources) * validation_fraction))
    test_count = max(1, int(len(sources) * test_fraction))
    if validation_count + test_count >= len(sources):
        raise ValueError('Split leaves no training source documents.')
    splits = {
        'val': sources[:validation_count],
        'test': sources[validation_count:validation_count + test_count],
        'train': sources[validation_count + test_count:],
    }
    counts = Counter(empty=empty_count, invalid=len(errors))
    if not dry_run:
        destinations = [output / f'{split}.txt' for split in splits]
        if any(destination.exists() for destination in destinations):
            raise FileExistsError('Split files already exist; refusing to overwrite them.')
        output.mkdir(parents=True, exist_ok=True)
    for split, source_ids in splits.items():
        lines = [line for source_id in source_ids for line in groups[source_id]]
        counts[split] = len(lines)
        if not dry_run:
            with (output / f'{split}.txt').open('w', encoding='utf-8', newline='') as label_file:
                label_file.writelines(lines)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description='Prepare normalized OpenOCR labels from CSV.')
    parser.add_argument('--config', default='./configs/configs/training.yaml')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-invalid', action='store_true')
    args = parser.parse_args()
    counts = prepare_labels(Config(args.config).cfg, args.dry_run, args.skip_invalid)
    print(('Validation only: ' if args.dry_run else 'Labels written: ') + str(counts))


if __name__ == '__main__':
    main()