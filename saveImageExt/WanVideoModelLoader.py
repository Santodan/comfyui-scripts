from ..meta import MetaField
from ..formatters import calc_model_hash, convert_skip_clip

SAMPLERS = {
    # WanVideoModelLoader is not a sampler
}

CAPTURE_FIELD_LIST = {
    "WanVideoModelLoader": {
        # Model
        MetaField.MODEL_NAME: {"field_name": "model"},
        MetaField.MODEL_HASH: {"field_name": "model", "format": calc_model_hash},

        # Clip skip
        MetaField.CLIP_SKIP: {"field_name": "clip_skip", "format": convert_skip_clip},

        # Prompts (only if your node forwards them)
        MetaField.POSITIVE_PROMPT: {"field_name": "positive"},
        MetaField.NEGATIVE_PROMPT: {"field_name": "negative"},
    },
}