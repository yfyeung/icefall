import json
import os
from collections import defaultdict

from datasets import load_dataset

splits = ["holdout", "dev", "train_base", "train_scaled"]

os.makedirs("data/manifests", exist_ok=True)

for split in splits:
    print(f"Processing split: {split}")

    ds = load_dataset("ajd12342/paraspeechcaps", split=split)

    data2sample = defaultdict(list)
    for sample in ds:
        data2sample[sample["source"]].append(sample)

    for source, samples in data2sample.items():
        output_path = os.path.join(
            "data/manifests", f"paraspeechcaps_{split}-{source}.jsonl"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                if source == "voxceleb":
                    audio_path = sample["relative_audio_path"].replace(
                        "_voicefixer", ""
                    )
                elif source == "expresso":
                    audio_path = os.path.join("expresso", sample["relative_audio_path"])
                elif source == "ears":
                    audio_path = os.path.join("ears", sample["relative_audio_path"])
                elif source == "emilia":
                    audio_path = os.path.join("Emilia", sample["relative_audio_path"])
                else:
                    raise ValueError

                audio_path = os.path.join("download", audio_path)
                text = sample["transcription"]
                caption = sample["text_description"]

                obj = {
                    "audio_path": audio_path,
                    "text": text,
                    "caption": caption,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(split, source, len(samples), "->", output_path)
