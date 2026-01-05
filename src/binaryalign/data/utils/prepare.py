import os
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict


def read_sentences(path: str) -> list[list[str]]:
    sentences = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Get sentence ID and text
            match = re.search(r"<s snum=(\d+)>(.*?)</s>", line, re.DOTALL)
            sent_id = int(match.group(1))
            text = match.group(2)
            # -- Store text split into words
            sentences[sent_id] = text.split()

    return sentences


def read_alignments(path: str):
    alignments = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # -- Get sentence ID and alignment indices
            match = re.search(r"(\d+) (\d+) (\d+) (.)?", line, re.DOTALL)
            sent_id = int(match.group(1))
            src_idx = int(match.group(2)) - 1
            tgt_idx = int(match.group(3)) - 1
            conf = match.group(4)
            # -- If no sure/possible, assume sure
            if conf == "S" or conf is None:
                alignments[sent_id].append((src_idx, tgt_idx))

    return alignments


def save_alignment_jsonl(path, src_sentences, tgt_sentences, alignments):
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt, al in zip(src_sentences, tgt_sentences, alignments):
            record = {
                "src_words": src,
                "tgt_words": tgt,
                "alignments": [list(pair) for pair in al],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_alignment_jsonl(path):
    src_sentences = []
    tgt_sentences = []
    alignments = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            src_sentences.append(record["src_words"])
            tgt_sentences.append(record["tgt_words"])
            alignments.append([tuple(p) for p in record["alignments"]])

    return src_sentences, tgt_sentences, alignments

"""
Functions for loading sentences/alignments per-dataset to a common training format
"""
def load_snum(src_path: str, tgt_path: str, align_path: str):
    src_data = read_sentences(src_path)
    tgt_data = read_sentences(tgt_path)
    align_data = read_alignments(align_path)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for sent_id in src_data:
        src_sentences.append(src_data[sent_id])
        tgt_sentences.append(tgt_data[sent_id])
        alignments.append(align_data[sent_id])

    return src_sentences, tgt_sentences, alignments


def load_golden_collection(src_path: str, tgt_path: str, align_path: str):
    src_data = read_sentences(src_path)
    tgt_data = read_sentences(tgt_path)
    align_data = read_alignments(align_path)

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for sent_id in src_data:
        src_sentences.append(src_data[sent_id])
        tgt_sentences.append(tgt_data[sent_id])
        alignments.append(align_data[sent_id])

    return src_sentences, tgt_sentences, alignments


def load_czenali(align_path: str):
    tree = ET.parse(align_path)
    root = tree.getroot()

    src_sentences = []
    tgt_sentences = []
    alignments = []

    for elem in root.iter("s"):
        english = elem.find("english").text
        czech = elem.find("czech").text
        sure = elem.find("sure").text

        src_sentences.append(english.split())
        tgt_sentences.append(czech.split())
        alignments.append({
            tuple(map(int, pair.split("-")))
            for pair in sure.split()
        })

    return src_sentences, tgt_sentences, alignments


def load_kftt(src_path: str, tgt_path: str, align_path: str):
    pass


def main():
    paths = {
        "Hansards": {
            "src_paths": [
                "datasets/fr-en/Hansards/test/test.e",
                "datasets/fr-en/Hansards/trial/trial.e"
            ],
            "tgt_paths": [
                "datasets/fr-en/Hansards/test/test.f",
                "datasets/fr-en/Hansards/trial/trial.f"
            ],
            "align_paths": [
                "datasets/fr-en/Hansards/test/test.wa.nonullalign",
                "datasets/fr-en/Hansards/trial/trial.wa"
            ]
        }
    }

    for dataset in paths:
        src_paths = paths[dataset]["src_paths"]
        tgt_paths = paths[dataset]["tgt_paths"]
        align_paths = paths[dataset]["align_paths"]

        for src_path, tgt_path, align_path in zip(src_paths, tgt_paths, align_paths):
            src_sentences, tgt_sentences, alignments = load_hansards(src_path, tgt_path, align_path)
            print(f"src_sentences[0]: {src_sentences[0]}")
            print(f"tgt_sentences[0]: {tgt_sentences[0]}")
            print(f"alignments[0]: {alignments[0]}")

if __name__ == "__main__":
    main()
