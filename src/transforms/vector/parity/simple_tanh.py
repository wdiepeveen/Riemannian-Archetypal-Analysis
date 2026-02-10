import torch

from src.transforms.vector.parity import ParityVectorTransform

class SimpleTanhParityVectorTransform(ParityVectorTransform):
    def __init__(self, d, hidden_d=8, parity=0):
        r = d // 2 + (d % 2 if parity == 1 else 0)

        fc1 = torch.nn.Linear(r, hidden_d)
        fc2 = torch.nn.Linear(hidden_d, d - r)
        act = torch.nn.Tanh()

        nn = torch.nn.Sequential(
            fc1,
            act,
            fc2,
        )
        super().__init__(d, r, nn, parity)

