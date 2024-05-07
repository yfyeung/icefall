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
from pathlib import Path

import jsonlines
from jiwer import cer
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recogs-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cuts-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--valids-file",
        type=str,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
    )

    return parser.parse_args()


def get_valids(args):
    with open(args.recogs_file) as f:
        lines = f.read().splitlines()
        lines = iter(lines)

    tot_cnt = 0
    val_cnt = 0
    val_ids = []
    while True:
        try:
            ref_id, ref = next(lines).split("\t")
            hyp_id, hyp = next(lines).split("\t")

            ref_id = ref_id.replace(":", "")
            hyp_id = hyp_id.replace(":", "")

            assert ref_id == hyp_id, f"{ref_id}, {hyp_id}"

            ref = (
                ref.replace("ref=[", "")
                .replace("]", "")
                .replace("'", "")
                .replace(",", "")
                .replace(" ", "")
            )
            hyp = (
                hyp.replace("hyp=[", "")
                .replace("]", "")
                .replace("'", "")
                .replace(",", "")
                .replace(" ", "")
            )
            score = cer(ref, hyp)

            tot_cnt += 1
            if score <= args.threshold:
                val_ids.append(ref_id)
                val_cnt += 1
        except StopIteration:
            break

    val_rate = val_cnt / tot_cnt
    logging.info(
        f"total cuts: {tot_cnt}, filtered cuts: {tot_cnt - val_cnt}, filtered rate: {1 - val_rate}"
    )

    with open("valids", "w") as f:
        for val_id in val_ids:
            f.write(val_id + "\n")


def get_valjsonl(args):
    with open(args.valids_file) as f:
        val_ids = f.read().splitlines()

    with jsonlines.open(args.cuts_file) as reader, jsonlines.open(
        args.cuts_file.replace(".jsonl", f"_threshold{args.threshold}.jsonl"), "w"
    ) as writer:
        for line in tqdm(reader):
            if line["id"] in val_ids[:10]:
                writer.write(line)
                val_ids.remove(line["id"])


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()

    if args.valids_file is None:
        get_valids(args)
    else:
        get_valjsonl(args)


if __name__ == "__main__":
    main()
