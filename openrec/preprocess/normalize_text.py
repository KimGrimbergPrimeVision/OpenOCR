import unicodedata

import yaml


class NormalizeLabel:

    def __init__(self, config_path, character_dict_path, use_space_char=False,
                 max_text_length=125, **kwargs):
        with open(config_path, encoding='utf-8') as config_file:
            self.config = yaml.safe_load(config_file)
        with open(character_dict_path, encoding='utf-8') as dictionary_file:
            characters = dictionary_file.read().splitlines()
        if any(len(character) != 1 for character in characters):
            raise ValueError('The dictionary must contain one character per line.')
        if len(set(characters)) != len(characters):
            raise ValueError('The dictionary contains duplicate characters.')
        self.characters = set(characters)
        if use_space_char:
            self.characters.add(' ')
        self.max_text_length = max_text_length
        punctuation = set(self.config.get('allowed_punctuation', []))
        missing = punctuation - self.characters
        if missing:
            raise ValueError(f'Punctuation missing from dictionary: {sorted(missing)}')
        if self.config.get('preserve_spaces', True) and ' ' not in self.characters:
            raise ValueError('preserve_spaces requires use_space_char=True.')

    def __call__(self, data):
        text = unicodedata.normalize('NFC', data['label'])
        if self.config.get('lowercase', False):
            text = text.lower()
        if self.config.get('normalize_diacritics', False):
            replacements = self.config.get('diacritic_map', {})
            text = ''.join(replacements.get(character, character)
                           for character in text)
            text = ''.join(character for character in unicodedata.normalize('NFD', text)
                           if not unicodedata.combining(character))
            text = unicodedata.normalize('NFC', text)
        if self.config.get('preserve_spaces', True):
            text = ' '.join(text.split())
        else:
            text = ''.join(text.split())
        unknown = set(text) - self.characters
        if unknown and not self.config.get('strip_unknown_characters', False):
            raise ValueError(f'Label contains unknown characters: {sorted(unknown)}')
        text = ''.join(character for character in text if character in self.characters)
        if not text or len(text) > self.max_text_length:
            raise ValueError(f'Normalized label length must be 1..{self.max_text_length}, '
                             f'got {len(text)}.')
        data['label'] = text
        return data