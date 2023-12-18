import torch

from nef import get_neural_field


class BaseModulationField(torch.nn.Module):

    def __init__(self, cfg, num_in, num_out, num_signals):
        super().__init__()

        # Store model parameters.
        self.cfg = cfg
        self.num_in = num_in
        self.num_out = num_out
        self.num_signals = num_signals

    def forward(self, x, patient_idx):
        raise NotImplementedError("Modulation field forward pass needs to be implemented.")


class NeuralModulationField(BaseModulationField):

    def __init__(self, cfg, num_in, num_out, num_signals):
        super().__init__(cfg, num_in, num_out, num_signals)

        # Instantiate the networks mapping to the conditioning variables.
        self.neural_modulation_fields = torch.nn.ModuleList(
            [get_neural_field(
                nef_cfg=self.cfg.neural_modulation_field,
                num_in=self.num_in,
                num_out=num_out,
            ) for _ in range(num_signals)]
        )

        if cfg.zero_init:
            # We want to initialize the neural modulation fields in such a way that beta = 0 gamma = 1 at init. The final
            # layer of the neural modulation field is a linear layer that maps to a vector of size [gamma + beta].
            for mod_field in self.neural_modulation_fields:
                for name, module in mod_field.named_modules():
                    if isinstance(module, torch.nn.Linear) and "final_linear" in name:
                        # Set weight to 0
                        module.weight.data.fill_(0.)

                        # Set bias to 0 for gamma and 1 for beta, so half of the bias vector is 1 and the other half 0.
                        # The bias is [gamma + beta], so we need to set the first half to 1 and the second half to 0.
                        module.bias.data.fill_(0.)
                        module.bias.data[:module.bias.data.shape[0]//2] = 1.

    def forward(self, coords, patient_idx):
        # Obtain modulations
        modulations = self.neural_modulation_fields[patient_idx](coords)
        return modulations


class CodeModulationField(BaseModulationField):
    def __init__(self, cfg, num_in, num_out, num_signals, code_dim):
        super().__init__(cfg, num_in, num_out, num_signals)

        # Instantiate a single neural modulation field that maps to the conditioning variables.
        self.conditioning = torch.nn.ModuleDict()

        # Instantiate a neural modulation field
        self.conditioning["embedding"] = get_neural_field(
            nef_cfg=self.cfg.code.coord_embedding,
            num_in=num_in,
            num_out=code_dim,
        )
        self.conditioning["neural_modulation_field"] = get_neural_field(
            nef_cfg=self.cfg.neural_modulation_field,
            num_in=2 * code_dim,
            num_out=num_out,
        )
        # Instantiate a code embedding layer
        self.conditioning["code"] = torch.nn.Embedding(
            num_embeddings=num_signals,
            embedding_dim=code_dim,
        )
        # Initialize embedding to be std [-1, 1]
        self.conditioning["code"].weight.data.uniform_(-1, 1)

        if cfg.zero_init:
            # Set weight to ~0
            self.conditioning.neural_modulation_field.final_linear.weight.data.normal_(0., 1e-5)

            # Set bias to 0 for gamma and 1 for beta, so half of the bias vector is 1 and the other half 0.
            # The bias is [gamma + beta], so we need to set the first half to 1 and the second half to 0.
            self.conditioning.neural_modulation_field.final_linear.bias.data.fill_(0.)
            self.conditioning.neural_modulation_field.final_linear.bias.data[
                :self.conditioning.neural_modulation_field.final_linear.bias.data.shape[0]//2] = 1.

    def forward(self, coords, patient_idx):
        # Obtain code for patient
        patient_idx = torch.tensor(patient_idx, dtype=torch.long, device=coords.device)
        code = self.conditioning.code(patient_idx)

        # Embed the coordinates
        emb_coords = self.conditioning.embedding(coords)

        # Repeat for all coordinates in the batch  [code_dim] -> [*batch_size, code_dim]
        code = code.view(*(1 for _ in coords.shape[:-1]), -1).repeat(*coords.shape[:-1], 1)

        # Concatenate code to coords
        mod_inp = torch.cat([emb_coords, code], dim=-1)

        # Obtain modulations
        modulations = self.conditioning.neural_modulation_field(mod_inp)  # [batch_size, 2 * num_layers * num_weights]
        return modulations


class CodeModulation(BaseModulationField):

    def __init__(self, cfg, num_in, num_out, num_signals, code_dim):
        super().__init__(cfg, num_in, num_out, num_signals)

        # Instantiate the networks mapping to the conditioning variables.
        self.conditioning = torch.nn.Embedding(
            num_embeddings=num_signals,
            embedding_dim=code_dim,
        )

        # Mapping from code to modulation
        self.codes = torch.nn.Linear(
            in_features=code_dim,
            out_features=num_out,
        )

        if cfg.zero_init:
            # Set weight to ~0
            self.codes.weight.data.uniform_(-1e-5, 1e-5)

    def forward(self, coords, patient_idx):
        patient_idx = torch.tensor(patient_idx, dtype=torch.long, device=coords.device)
        code = self.conditioning(patient_idx)

        # Map code to modulation
        modulations = self.codes(code)

        # Reshape the code to [*batch_size, code_dim]
        modulations = modulations.view(*(1 for _ in coords.shape[:-1]), -1).repeat(*coords.shape[:-1], 1)
        return modulations
