import torch


def get_optimizer(optimizer_config, parameters, lr, **kwargs):
    # Optimizer
    if optimizer_config.type == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            weight_decay=optimizer_config.weight_decay,
            lr=lr,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
            **kwargs,
        )
    elif optimizer_config.type == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            weight_decay=optimizer_config.weight_decay,
            lr=lr,
            momentum=optimizer_config.momentum,
            nesterov=optimizer_config.nesterov,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_config.type} not implemented.")
    return optimizer
