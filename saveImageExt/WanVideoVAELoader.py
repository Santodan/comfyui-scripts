from ..meta import MetaField
from ..formatters import calc_vae_hash

SAMPLERS = {
    # WanVideoVAELoader is not a sampler
}

CAPTURE_FIELD_LIST = {
    "WanVideoVAELoader": {
        # VAE model
        MetaField.VAE_NAME: {"field_name": "model_name"},
        MetaField.VAE_HASH: {"field_name": "model_name", "format": calc_vae_hash},

        # Optional: include precision if you want to log it
        # MetaField.PRECISION: {"field_name": "precision"},
    },
}