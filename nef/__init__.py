from functools import partial

from nef.mlp import MLP
from nef.hashnet import HashNet
from nef.rffnet import RFFNet
from nef.siren import Siren


def get_neural_field(nef_cfg, num_in, num_out):
    """Get neural field model as specified by config.

    Args:
        cfg: Config object.
        num_in: Number of input features (dimensionality of volume to reconstruct).
        num_signals: Number of signals (used for conditional nef).
    """
    if nef_cfg.type == "MLP":
        nef = MLP(
            num_in=num_in,
            num_hidden_in=nef_cfg.num_hidden,
            num_hidden_out=nef_cfg.num_hidden,
            num_layers=nef_cfg.num_layers,
            num_out=num_out,
            final_act=nef_cfg.final_act,
        )
    elif nef_cfg.type == "Hash":
        nef = HashNet(
            num_in=num_in,
            num_hidden_in=nef_cfg.num_hidden,
            num_hidden_out=nef_cfg.num_hidden,
            num_layers=nef_cfg.num_layers,
            num_out=num_out,
            num_levels=nef_cfg.hash.num_levels,
            level_dim=nef_cfg.hash.level_dim,
            base_resolution=nef_cfg.hash.base_resolution,
            log2_max_params_per_level=nef_cfg.hash.log2_max_params_per_level,
            final_act=nef_cfg.final_act,
            skip_conn=nef_cfg.hash.skip_conn,
        )
    elif nef_cfg.type == "RFF":
        nef = RFFNet(
            num_in=num_in,
            num_hidden_in=nef_cfg.num_hidden,
            num_hidden_out=nef_cfg.num_hidden,
            num_layers=nef_cfg.num_layers,
            num_out=num_out,
            std=nef_cfg.rff.std,
            learnable_coefficients=nef_cfg.rff.learnable_coefficients,
            final_act=nef_cfg.final_act,
        )
    elif nef_cfg.type == "SIREN":
        nef = Siren(
            num_in=num_in,
            num_hidden_in=nef_cfg.num_hidden,
            num_hidden_out=nef_cfg.num_hidden,
            num_layers=nef_cfg.num_layers,
            num_out=num_out,
            omega=nef_cfg.SIREN.omega,
            final_act=nef_cfg.final_act,
        )
    else:
        raise NotImplementedError(f"Unrecognized model type: {nef_cfg.type}.")

    return nef
