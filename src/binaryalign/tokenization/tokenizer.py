import torch
from transformers import AutoTokenizer, BatchEncoding


class AlignTokenizer:
    def __init__(self, model_name: str, markers=["<ws>","</ws>"]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ----------
        # Add source work marker tokens / resize embeddings
        # ----------
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": markers}
        )
        self.word_start, self.word_end = markers[0], markers[1]
        self.embedding_size = len(self.tokenizer)

    def encode_marked_pair(
            self, 
            src_words: list[str], 
            tgt_words: list[str],
            src_word_idx: int
    ):
        """
        
        """
        marked_src = self.mark_source_word(src_words, src_word_idx)
        encoding = self.encode_pair(marked_src, tgt_words)
        tgt_masks = self.get_target_masks(encoding, 0)

        return encoding, tgt_masks

    def mark_source_word(self, src_words: list[str], i: int) -> list[str]:
        """

        """
        marked_src = src_words[:i] + [self.word_start, src_words[i], self.word_end] + src_words[i+1:]
        return marked_src
    
    def encode_pair(self, src_words: list[str], tgt_words: list[str]) -> BatchEncoding:
        """
        
        """
        return self.tokenizer(
            src_words,
            tgt_words,
            is_split_into_words=True,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
    
    def get_target_masks(self, encoding: BatchEncoding, batch_idx: int):
        """
        
        """
        seq_ids = encoding.sequence_ids(batch_idx)
        tgt_mask = torch.tensor(
            [seq_id == 1 for seq_id in seq_ids],
            dtype=torch.bool
        )
        return tgt_mask
    
    def get_tokens(self, encoding: BatchEncoding, batch_idx: int):
        """
        
        """
        ids = encoding["input_ids"][batch_idx]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        return tokens


def main():
    tokenizer = AlignTokenizer("microsoft/mdeberta-v3-base")

    src_words = ["unbelievable", "results"]
    tgt_words = ["des", "resultats", "incroyables"]

    src_words = tokenizer.mark_source_word(src_words, 0)
    encoding = tokenizer.encode_pair(src_words, tgt_words)
    tokens = tokenizer.get_tokens(encoding, 0)
    print(f"tokens: {tokens}")
    tgt_masks = tokenizer.get_target_masks(encoding, 0)

    print(f"tgt_masks: {tgt_masks}")


if __name__ == "__main__":
    main()