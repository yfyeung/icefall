#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


"""
This file computes fbank features of the musan dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/manifests.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    MonoCut,
    WhisperFbank,
    WhisperFbankConfig,
    combine,
)
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def is_cut_long(c: MonoCut) -> bool:
    return c.duration > 5


def compute_fbank_musan(
    num_mel_bins: int = 80,
    whisper_fbank: bool = False,
    output_dir: str = "data/manifests",
):
    src_dir = Path("data/manifests")
    output_dir = Path(output_dir)
    num_jobs = min(15, os.cpu_count())

    dataset_parts = (
        "music",
        "speech",
        "noise",
    )
    prefix = "musan"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    musan_cuts_path = output_dir / "musan_cuts.jsonl.gz"

    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for Musan")

    if whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=num_mel_bins, device="cuda")
        )
    else:
        extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        # create chunks of Musan with duration 5 - 10 seconds
        musan_cuts = (
            CutSet.from_manifests(
                recordings=combine(part["recordings"] for part in manifests.values())
            )
            .cut_into_windows(10.0)
            .filter(is_cut_long)
            .compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/musan_feats",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
        )
        musan_cuts.to_file(musan_cuts_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bins for Fbank""",
    )
    parser.add_argument(
        "--whisper-fbank",
        type=str2bool,
        default=False,
        help="Use WhisperFbank instead of Fbank. Default: False.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/manifests",
        help="Output directory. Default: data/manifests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    compute_fbank_musan(
        num_mel_bins=args.num_mel_bins,
        whisper_fbank=args.whisper_fbank,
        output_dir=args.output_dir,
    )
