import logging
from functools import lru_cache
from uuid import UUID

from huggingface_hub import snapshot_download

from common.config import Settings, get_settings
from common.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingType,
    TextEmbedding,
)
from common.tokenization import TextValidator, load_tokenizer

logger = logging.getLogger(__name__)


def resolve_device(settings: Settings, torch) -> str:
    if settings.inference_device == "cpu":
        return "cpu"
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "INFERENCE_DEVICE=amd/amd-wsl requires a ROCm build of PyTorch "
            "and an accessible AMD GPU"
        )
    # ROCm PyTorch intentionally exposes devices through the torch.cuda API.
    return "cuda:0"


class InferenceEngine:
    def __init__(self, settings: Settings):
        # Heavy imports stay exclusively in the worker.
        import torch
        from FlagEmbedding import BGEM3FlagModel

        self.settings = settings
        device = resolve_device(settings, torch)
        logger.info("Loading %s on %s", settings.model_id, settings.inference_device)
        path = snapshot_download(
            repo_id=settings.model_id,
            revision=settings.model_revision,
            allow_patterns=[
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "sentencepiece.bpe.model",
                "pytorch_model.bin",
                "model.safetensors",
                "colbert_linear.pt",
                "sparse_linear.pt",
            ],
        )
        self.model = BGEM3FlagModel(
            path,
            devices=[device],
            use_fp16=settings.use_fp16 and device != "cpu",
            normalize_embeddings=True,
        )
        self.validator = TextValidator(load_tokenizer(settings), settings)
        logger.info("BGE-M3 is ready")

    def encode(self, task_id: str, request: EmbeddingRequest) -> EmbeddingResult:
        self.validator.validate(request.input)
        selected = set(request.embedding_types)
        data = []
        if request.input:
            output = self.model.encode(
                request.input,
                batch_size=self.settings.inference_batch_size,
                max_length=self.settings.max_input_tokens,
                return_dense=EmbeddingType.DENSE in selected,
                return_sparse=EmbeddingType.SPARSE in selected,
                return_colbert_vecs=EmbeddingType.COLBERT in selected,
            )
            for index in range(len(request.input)):
                item = TextEmbedding(index=index)
                if EmbeddingType.DENSE in selected:
                    item.embedding = output["dense_vecs"][index].tolist()
                if EmbeddingType.SPARSE in selected:
                    item.sparse = {
                        str(token_id): float(weight)
                        for token_id, weight in output["lexical_weights"][index].items()
                    }
                if EmbeddingType.COLBERT in selected:
                    item.colbert = output["colbert_vecs"][index].tolist()
                data.append(item)
        return EmbeddingResult(
            task_id=UUID(task_id),
            model=self.settings.model_id,
            embedding_types=request.embedding_types,
            data=data,
        )


@lru_cache(maxsize=1)
def get_engine() -> InferenceEngine:
    return InferenceEngine(get_settings())
