"""WaveLM — a complex wave-field byte-level language model.

See the master plan (spec/wave_lm/) for the full mathematical specification.
"""

from .config import WaveConfig, wave_micro_config, wave_base_config
from .field import (
    causal_dilated_conv, saturating_nonlinearity, phase_rotation,
    normalize_field, wave_step, channel_mix, channel_mix_lowrank, selective_gate,
)
from .readout import born_logits, linear_logits, ce_loss, bits_per_byte
from .model import WaveModel
from .real_model import RealWaveModel

__all__ = [
    "WaveConfig",
    "wave_micro_config",
    "wave_base_config",
    "causal_dilated_conv",
    "saturating_nonlinearity",
    "phase_rotation",
    "normalize_field",
    "wave_step",
    "channel_mix",
    "channel_mix_lowrank",
    "selective_gate",
    "born_logits",
    "linear_logits",
    "ce_loss",
    "bits_per_byte",
    "WaveModel",
    "RealWaveModel",
]
