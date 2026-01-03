from transformers import AutoModel


def load_backbone(model_name: str) -> AutoModel:
    return AutoModel.from_pretrained(model_name)