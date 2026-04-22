import torch

class RiemannianNeuralNetwork(torch.nn.Module):
    def __init__(self, manifold):
        super().__init__()
        self.d = manifold.d
        self.manifold = manifold

    def forward(self, x):
        raise NotImplementedError("This method should be implemented by subclasses.")