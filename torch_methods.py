import torch
import torchphysics as tp


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call = torch.nn.Sequential(
            torch.nn.Linear(2, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.call(x)


def load_torch_model(name: str):

    X = tp.spaces.R2('x')
    U = tp.spaces.R1('u')
    tmp_model = tp.models.FCN(input_space=X,
                              output_space=U,
                              hidden=(50, 50, 50),
                              activations=torch.nn.Tanh(),
                              xavier_gains=1.0)

    tmp_model.load_state_dict(torch.load('./PINN_pde_solutions/' + name))

    return tmp_model.sequential
