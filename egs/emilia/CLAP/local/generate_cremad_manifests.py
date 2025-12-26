import argparse
import glob
import json
import logging
import os

import torch
from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

EMOTIONS = [
    "A",
    "D",
    "F",
    "H",
    "N",
    "S",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the cremad dataset",
        default="./download/cremad",
    )

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_dir = args.manifest_dir
    os.makedirs(manifest_dir, exist_ok=True)

    for split in [
        "test",
        # "valid",
        # "train",
    ]:
        dataset = {}

        label_paths = sorted(glob.glob(f"{dataset_dir}/{split}/*.json"))
        for label_path in label_paths:
            with open(label_path, "r") as f:
                item = json.load(f)
            emotion = item["label"]

            audio_path = label_path.replace(".json", ".wav")
            assert os.path.isfile(audio_path)
            audio_name = audio_path.split("/", 1)[-1].replace(".wav", "")

            dataset[audio_name] = [audio_path, emotion]

        logging.info(f"A total of {len(dataset)} clips!")

        cuts = []
        for i, (cut_id, info) in enumerate(dataset.items()):
            audio_path, emotion = info
            recording = Recording.from_file(audio_path, cut_id)
            cut = MonoCut(
                id=cut_id,
                start=0,
                duration=recording.duration,
                channel=0,
                recording=recording,
            )
            supervision = SupervisionSegment(
                id=cut_id,
                recording_id=cut.recording.id,
                start=0,
                channel=0,
                duration=cut.duration,
                text="",
            )
            supervision.emotion = emotion

            cut.supervisions = [supervision]
            cut = cut.resample(16000)

            cuts.append(cut)

            if i % 100 == 0 and i:
                logging.info(f"Processed {i} cuts until now.")

        cuts = CutSet.from_cuts(cuts)

        manifest_output_dir = manifest_dir + "/" + f"cremad_cuts_{split}.jsonl.gz"

        logging.info(f"Storing the manifest to {manifest_output_dir}")
        cuts.to_jsonl(manifest_output_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
