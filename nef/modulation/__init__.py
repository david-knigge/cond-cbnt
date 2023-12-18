from nef.modulation.modulator import BaseNeuralFieldModulator
from nef import get_neural_field


def get_modulated_neural_field(nef_cfg, num_in, num_out, num_signals=None):

    nef = get_neural_field(nef_cfg, num_in, num_out)

    # Next, we instantiate the conditioner
    conditioner = BaseNeuralFieldModulator(
        nef=nef,
        num_in=num_in,
        num_signals=num_signals,
        conditioning_cfg=nef_cfg.conditioning,
    )

    return conditioner