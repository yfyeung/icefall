#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 ./local/compute_neural_codec_and_prepare_text_tokens.py --dataset-parts "${dataset_parts}" \
        --text-extractor ${text_extractor} \
        --audio-extractor ${audio_extractor} \
        --batch-duration 2500 --prefix "wenetspeech4tts" \
        --src-dir "data/manifests" --split 100 \
        --output-dir "${audio_feats_dir}/wenetspeech4tts_${dataset_parts}_split_100"

"""
import argparse
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.multiprocessing
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.features import FeatureExtractor
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from tqdm.auto import tqdm

from icefall.utils import get_executor

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


import re
from typing import Pattern

import numpy as np
from k2 import SymbolTable

# from valle.data import (
#     AudioTokenConfig,
#     AudioTokenExtractor,
#     TextTokenizer,
#     tokenize_text,
# )
# from valle.data.fbank import get_fbank_extractor
# from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="dev-clean test-clean",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="libritts",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="Split the cut_set into multiple parts",
    )

    return parser.parse_args()


class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(_text, style=Style.TONE3, neutral_tone_with_five=True)
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = get_finals(py[0][:-1], strict=False) + py[0][-1]
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone] + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


def main():
    args = get_args()

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )

    text_tokenizer = None
    if args.text_extractor:
        text_tokenizer = TextTokenizer(backend=args.text_extractor)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    unique_symbols = set()
    num_jobs = min(32, os.cpu_count())
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    with get_executor() as ex:
        for partition, m in manifests.items():
            logging.info(
                f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
            )
            try:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )
            except Exception:
                cut_set = m["cuts"]

            # Split cut_set if split > 1
            split = 1
            if args.split > 1:
                cut_sets = cut_set.split(args.split)
                split = args.split
            else:
                cut_sets = [cut_set]

            for idx, part in enumerate(cut_sets):
                # TextTokenizer
                if args.text_extractor:
                    for c in tqdm(part):
                        if args.prefix in [
                            "aishell",
                            "aishell2",
                            "wenetspeech4tts",
                            "libritts",
                            "libritts-r",
                        ]:
                            phonemes = tokenize_text(
                                text_tokenizer, text=c.supervisions[0].text
                            )
                            if c.supervisions[0].custom is None:
                                c.supervisions[0].custom = {}
                            c.supervisions[0].normalized_text = c.supervisions[0].text
                        else:
                            raise NotImplementedError(f"{args.prefix}")
                        unique_symbols.update(phonemes)
                        c.tokens = phonemes
                        assert c.supervisions[
                            0
                        ].normalized_text, "normalized_text is None"

                # Save each part with an index if split > 1
                if split > 1:
                    cuts_filename = f"{prefix}cuts_{partition}.{idx}.{args.suffix}"
                else:
                    cuts_filename = f"{prefix}cuts_{partition}.{args.suffix}"

                part.to_file(f"{args.output_dir}/{cuts_filename}")
                logging.info(f"Saved {cuts_filename}")

    if args.text_extractor:
        unique_phonemes = SymbolTable()
        for s in sorted(list(unique_symbols)):
            unique_phonemes.add(s)
        logging.info(f"{len(unique_symbols)} unique phonemes: {unique_symbols}")

        unique_phonemes_file = f"{args.output_dir}/unique_text_tokens.k2symbols"
        unique_phonemes.to_file(unique_phonemes_file)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
