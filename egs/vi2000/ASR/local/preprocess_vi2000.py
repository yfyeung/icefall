#!/usr/bin/env python3

import logging
import unicodedata
from pathlib import Path

import regex as re
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

VN_EN_RE = re.compile(
    r"^(?:[a-zđ](?:[\u0300-\u0303\u0306\u0309\u031b\u0323])*|\s)+$",
    re.IGNORECASE,
)
PUNC_SYM_RE = re.compile(r"[\p{P}\p{N}]+", re.VERSION1)


def is_vietnamese(sup: SupervisionSegment) -> bool:
    text = unicodedata.normalize("NFD", sup.text)
    text = PUNC_SYM_RE.sub("", text)
    return bool(VN_EN_RE.fullmatch(text))


def normalize_text(
    text: str,
) -> str:
    # unicode NFC normalization for Vietnamese diacritics
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\p{Mn}+", "", text)

    # remove punctuation and symbols
    text = re.sub(r"[\p{P}\p{S}]", "", text)

    # lowercase
    text = text.lower()

    # normalize multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_vi2000():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = ("test", "dev", "train")

    logging.info("Loading manifest")
    prefix = "vi2000"
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
        m["supervisions"] = m["supervisions"].filter(is_vietnamese)

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
        # if partition not in ["dev", "test"]:
        #     logging.info(f"Speed perturb for {partition} with factors 0.9 and 1.1 ")
        #     cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess_vi2000()


if __name__ == "__main__":
    main()
