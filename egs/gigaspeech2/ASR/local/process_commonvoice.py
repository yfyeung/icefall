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
import string
import unicodedata
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    return parser.parse_args()


def normalize_text(
    text: str,
    lang: str,
) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Convert to upper case
    text = text.upper()

    # Remove brackets with content
    text = re.sub(r"\([^\)]*\)", " ", text)

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s", "", text)

    return text


def preprocess_commonvoice(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    # number of workers in dataloader
    num_workers = 20

    # number of seconds in a batch
    batch_duration = 1000

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    dataset_parts = args.dataset.strip().split(" ", -1)

    logging.info("Loading manifest (may take 4 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=f"cv-{args.lang}",
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
        cuts_path = output_dir / f"cv-{args.lang}_cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        for sup in m["supervisions"]:
            old_text = sup.text
            new_text = normalize_text(old_text, args.lang)
            if old_text != new_text:
                logging.info(f"\nraw: {old_text}\nnew: {new_text}")
            sup.text = new_text

        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        ).resample(16000)

        logging.info("Computing features")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/cv-{args.lang}_feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            overwrite=True,
        )
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    preprocess_commonvoice(args)


if __name__ == "__main__":
    main()
