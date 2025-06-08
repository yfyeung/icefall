#!/usr/bin/env python3

import logging
from pathlib import Path

import regex as re
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached


def is_arabic_only(sup: SupervisionSegment) -> bool:
    text = sup.text
    if not re.search(r"\p{Arabic}", text):
        return False
    patch_pattern = r"\u0640\u064B\u064D\u064E\u064F\u0650\u0651\u0652"
    allowed_pattern = rf"[\p{{Arabic}}{patch_pattern}\p{{P}}\p{{S}}\p{{Z}}]+"
    remaining = re.sub(allowed_pattern, "", text)
    return len(remaining) == 0


def normalize_text(
    text: str,
) -> str:
    """
    Arabic text normalization:
    https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/blob/main/eval.py
    """
    # Remove punctuation and symbols
    text = re.sub(r"[\p{P}\p{S}]", "", text)

    # Remove diacritics
    diacritics = r"[\u064B-\u0652]"  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, "", text)

    # Remove Tatweel (Kashida, U+0640)
    text = re.sub(r"\u0640", "", text)

    # Normalize multiple whitespace characters into a single space
    text = re.sub(r"\s\s+", " ", text)

    return text.strip()


def preprocess_dataoceanai_arabic():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = ("train", "test")

    logging.info("Loading manifest")
    prefix = "dataoceanai-alg"
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

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"{prefix}_cuts_{partition}_raw.{suffix}"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        # We must filter out some utterances and remove punctuation
        logging.info("Filtering utterances from supervisions")
        m["supervisions"] = m["supervisions"].filter(is_arabic_only)

        logging.info(f"Normalizing text in {partition}")
        for sup in m["supervisions"]:
            sup.text = normalize_text(sup.text)

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        if partition not in ["dev", "test"]:
            logging.info(f"Speed perturb for {partition} with factors 0.9 and 1.1 ")
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess_dataoceanai_arabic()


if __name__ == "__main__":
    main()
