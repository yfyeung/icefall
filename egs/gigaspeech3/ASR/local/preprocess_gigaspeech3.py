#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.             (Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import argparse
import logging
import re
from pathlib import Path

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached
from num2words import num2words

from icefall.utils import str2bool


def normalize_text(text: str) -> str:
    # Convert to upper case
    text = text.upper()

    # Remove brackets with content
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    def _num2words(match):
        number = int(match.group())
        return num2words(number, lang="ko")

    # Replace numbers with Korean words
    text = re.sub(r"\d+", _num2words, text)

    # Remove blank symbols
    text = re.sub(r"\s+", "", text).strip()

    return text


def validate_text(text: str) -> bool:
    pattern = r"^[\u1100-\u11FF\uAC00-\uD7AFA-Z]+$"
    return bool(re.match(pattern, text))


def preprocess_gigaspeech3():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = (
        "test",
        "ytnnews24-0",
        "ytnnews24-1",
    )

    logging.info("Loading manifest")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix="gigaspeech3",
        suffix="jsonl.gz",
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"gigaspeech3_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        logging.info(f"Normalizing text in {partition}")
        for sup in m["supervisions"]:
            sup.text = normalize_text(sup.text)
            assert validate_text(sup.text), sup.text

        # Create cut manifests.
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    preprocess_gigaspeech3()


if __name__ == "__main__":
    main()
