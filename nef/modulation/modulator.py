import functools as ft

import torch

from nef.modulation.modulations import CodeModulationField, NeuralModulationField
# from nef.modulation.modulations import NeuralModulationField


class BaseNeuralFieldModulator(torch.nn.Module):
    '''
    Base Conditioning Class
    film_conditioning_hook():
        Defines hook that is injected into self.neural_field
    forward(coords, data):
        First computes the conditioning variables from the data
        Then conditions forward pass of NeuralField with CondVars
    conditioning_variables():

    '''

    def __init__(self, nef, conditioning_cfg, num_in, num_signals):
        super().__init__()

        # Store cfg and nef
        self.nef = nef
        self.num_in = num_in
        self.num_signals = num_signals
        self.cfg = conditioning_cfg

        # Set warmup config
        self.warmup_steps = self.cfg.warmup_steps
        self.step = 0

        # Extract relevant linear layers out of neural field as well as their dimension.
        self.nef_linear_layers = []
        for name, module in self.nef.hidden_layers.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.nef_linear_layers.append(module)

        # Add final linear layer
        self.nef_linear_layers.append(self.nef.final_linear)

        # Keep track of the number of output channels for the linear layers in the nef
        self.nef_linear_out_dims = self.nef_linear_layers[0].in_features

        # Register forward hook for modulation.
        for i, lin_layer in enumerate(self.nef_linear_layers):
            lin_layer.register_forward_pre_hook(ft.partial(self.film_conditioning_hook, layer_idx=i))

        if self.cfg.type == 'neural_modulation_field':
            self.conditioning = NeuralModulationField(
                cfg=self.cfg,
                num_in=self.num_in,
                num_out=2 * self.nef_linear_out_dims * len(self.nef_linear_layers),
                num_signals=self.num_signals,
            )
        elif self.cfg.type == 'code_modulation_field':
            self.conditioning = CodeModulationField(
                cfg=self.cfg,
                num_in=self.num_in,
                num_out=2 * self.nef_linear_out_dims * len(self.nef_linear_layers),
                num_signals=self.num_signals,
                code_dim=self.cfg.code_dim,
            )

        # Create buffer to store modulations.
        self.gamma = None
        self.beta = None

    def apply_warmup_decay(self, gamma, beta):
        # Calculate linear decay for this step.
        warmup = self.step / self.warmup_steps  # increases from 0 to 1 linearly

        # Update conditioning variables.
        gamma_t = torch.ones_like(gamma).fill_(1 - warmup) + gamma * warmup  # Gamma starts at 1
        beta_t = torch.zeros_like(beta) + beta * warmup  # Beta starts at 0

        # Increase training step counter
        if self.training:
            self.step += 1

        return gamma_t, beta_t

    def forward(self, coords, patient_idx):

        # We force all coordinates in a batch to be from the same patient, so just select the first of the corresponding
        # indices. [batch_size, 2 * num_layers * num_weights]
        modulations = self.conditioning(coords, patient_idx[0])

        # Chunk into gamma and beta, 2 x [batch_size, num_layers * num_weights]
        gamma, beta = torch.chunk(modulations, 2, dim=-1)

        # Reshape modulation to [*batch_size, modulation_dim, num_layers]
        gamma = gamma.reshape(*coords.shape[:-1], self.nef_linear_out_dims, len(self.nef_linear_layers))
        beta = beta.reshape(*coords.shape[:-1], self.nef_linear_out_dims, len(self.nef_linear_layers))

        if self.warmup_steps > 0 and self.step <= self.warmup_steps:
            gamma, beta = self.apply_warmup_decay(gamma, beta)

        # Set modulations as attribute, to be used in the forward hooks.
        self.gamma, self.beta = gamma, beta

        # Apply forward
        return self.nef(coords)  # -> [BS, *spatial_dims, #num_weights]

    def film_conditioning_hook(
            self, linear_layer, pre_activation, layer_idx
    ):
        # Unpack args
        pre_activation = pre_activation[0]

        # Get gamma and beta modulations [batch_size, modulation_dim]
        gamma_i, beta_i = (torch.select(self.gamma, -1, layer_idx),
                           torch.select(self.beta, -1, layer_idx))

        # Apply conditioning to activations.
        if self.cfg.do:
            pre_activation = gamma_i * pre_activation + beta_i
            # pre_activation = pre_activation + (gamma_i * pre_activation + beta_i)

        # Pass conditioned activations to the next layer.
        return pre_activation
