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

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recogs-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def filter_gigaspeech2(args):
    with open(args.recogs_file) as f:
        lines = f.read().splitlines()
        lines = iter(lines)

    tot_cnt = 0
    val_cnt = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    cnt_4 = 0
    cnt_5 = 0
    cnt_6 = 0
    cnt_7 = 0
    cnt_8 = 0
    cnt_9 = 0
    cnt_10 = 0
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
            if score <= 1:
                cnt_10 += 1
            if score <= 0.9:
                cnt_9 += 1
            if score <= 0.8:
                cnt_8 += 1
            if score <= 0.7:
                cnt_7 += 1
            if score <= 0.6:
                cnt_6 += 1
            if score <= 0.5:
                cnt_5 += 1
            if score <= 0.4:
                cnt_4 += 1
            if score <= 0.3:
                cnt_3 += 1
            if score <= 0.2:
                cnt_2 += 1
            if score <= 0.1:
                cnt_1 += 1
        except StopIteration:
            break
     
    for i in range(1, 11):
        val_cnt = eval(f"cnt_{i}")
        val_rate = val_cnt / tot_cnt
        logging.info(
            f"total cuts: {tot_cnt}, threshold: {i / 10}, valid cuts: {val_cnt}, filtered rate: {1 - val_rate}"
        )


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    filter_gigaspeech2(args)


if __name__ == "__main__":
    main()
