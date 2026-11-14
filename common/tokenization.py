from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from common.config import Settings


class InputTooLong(ValueError):
    def __init__(self, indices: list[int], limit: int):
        self.indices = indices
        self.limit = limit
        super().__init__(
            f"Texts at indices {indices} exceed {limit} tokens (including special tokens)"
        )


def load_tokenizer(settings: Settings) -> Tokenizer:
    path = hf_hub_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        filename="tokenizer.json",
    )
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


class TextValidator:
    def __init__(self, tokenizer: Tokenizer, settings: Settings):
        self.tokenizer = tokenizer
        self.settings = settings

    def validate(self, texts: list[str]) -> None:
        if len(texts) > self.settings.max_texts:
            raise ValueError(f"At most {self.settings.max_texts} texts are allowed")
        # One text at a time avoids retaining all token IDs for a large batch.
        invalid = [
            i
            for i, text in enumerate(texts)
            if len(self.tokenizer.encode(text, add_special_tokens=True).ids)
            > self.settings.max_input_tokens
        ]
        if invalid:
            raise InputTooLong(invalid, self.settings.max_input_tokens)
