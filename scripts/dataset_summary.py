import argparse
import json
import os
from collections import Counter

import soundfile as sf


def folder_label(folder):
    parts = folder.split("_")
    if len(parts) < 4:
        return None
    return {
        "distance": parts[0],
        "height": parts[1],
        "azimuth": parts[2],
        "orientation_code": parts[3],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--divided-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw_folders = sorted(
        p for p in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, p))
    )
    output_wavs = []
    label_files = []
    total_raw_seconds = 0.0
    first_wav_info = None

    for folder in raw_folders:
        folder_path = os.path.join(args.raw_dir, folder)
        wav_path = os.path.join(folder_path, "output.wav")
        label_path = os.path.join(folder_path, "label.json")
        if os.path.exists(label_path):
            label_files.append(label_path)
        if os.path.exists(wav_path):
            output_wavs.append(wav_path)
            info = sf.info(wav_path)
            total_raw_seconds += info.frames / info.samplerate
            if first_wav_info is None:
                first_wav_info = {
                    "file": wav_path,
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype,
                    "duration_seconds": info.frames / info.samplerate,
                }

    divided = {}
    labels = {"train": Counter(), "test": Counter()}
    examples = {"train": [], "test": []}
    for split in ["train", "test"]:
        split_dir = os.path.join(args.divided_dir, split)
        folders = sorted(
            p for p in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, p))
        )
        wav_count = 0
        for folder in folders:
            wavs = [
                os.path.join(split_dir, folder, p)
                for p in os.listdir(os.path.join(split_dir, folder))
                if p.lower().endswith(".wav")
            ]
            wav_count += len(wavs)
            label = folder_label(folder)
            if label:
                labels[split]["distance=" + label["distance"]] += 1
                labels[split]["height=" + label["height"]] += 1
                labels[split]["azimuth=" + label["azimuth"]] += 1
                labels[split]["orientation=" + label["orientation_code"]] += 1
            if len(examples[split]) < 5 and wavs:
                examples[split].append(wavs[0])
        divided[split] = {"folders": len(folders), "wav_clips": wav_count}

    summary = {
        "raw_recording_folders": len(raw_folders),
        "raw_output_wavs": len(output_wavs),
        "raw_label_json_files": len(label_files),
        "raw_total_duration_seconds": total_raw_seconds,
        "raw_total_duration_hours": total_raw_seconds / 3600.0,
        "first_wav_info": first_wav_info,
        "divided_dataset": divided,
        "label_counts_by_split": {split: dict(counter) for split, counter in labels.items()},
        "example_files": examples,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
