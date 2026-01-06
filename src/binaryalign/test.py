import os
from pathlib import Path
import argparse
import spacy
import torch
from omegaconf import OmegaConf, DictConfig

from binaryalign.models import BinaryAlignClassifier, BinaryAlignModel, load_backbone
from binaryalign.tokenization import BinaryAlignTokenizer
from binaryalign.inference.align import BinaryAlign


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def main():
    # ----------
    # Parse arguments / load config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    test_config = load_config(args.config)

    train_dir = Path(test_config.run.run_dir) / "training"
    train_config = load_config(train_dir / "config.yml")

    # ---------
    # Create Testing Dirs / Save Config
    # ----------
    test_dir = Path(test_config.run.run_dir) / "testing" / test_config.run.name
    os.makedirs(test_dir, exist_ok=True)

    save_config(test_config, test_dir / "config.yml")

    # ----------
    # Load tokenizer/backbone
    # ----------
    tokenizer = BinaryAlignTokenizer(model_name=train_config.model.backbone, max_length=train_config.model.max_length)
    backbone = load_backbone(train_config.model.backbone, tokenizer.vocab_size)

    # ----------
    # Load BinaryAlignModel checkpoint
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = backbone.config.hidden_size

    classifier = BinaryAlignClassifier(hidden_dim)
    model = BinaryAlignModel(backbone, classifier)

    ckpt_path = train_dir / "checkpoints" / test_config.run.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # ----------
    # Create BinaryAlign inference model
    # ----------
    src_lang = test_config.inference.src.lang
    tgt_lang = test_config.inference.tgt.lang

    binaryalign = BinaryAlign(model, tokenizer, src_lang, tgt_lang)

    # ----------
    # Run inference on test samples
    # ----------
    threshold = test_config.inference.threshold

    src_sentences = test_config.inference.src.sentences
    tgt_sentences = test_config.inference.tgt.sentences

    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
        print("\n====================================")
        print(f"Source ({src_lang}): {src_sent}")
        print(f"Target ({tgt_lang}): {tgt_sent}\n")

        src_words, tgt_words, alignments = binaryalign.align(src_sent, tgt_sent, threshold)

        for (src_idx, src_word) in alignments:
            print(f"({src_idx}) {src_word}: {[(i, tgt_word, s) for i, tgt_word, s in alignments[(src_idx, src_word)]]}")

        print("====================================")

if __name__ == "__main__":
    main()