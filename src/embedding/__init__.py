import torch

class Embedding(torch.nn.Module):
    def __init__(self, d):
        super(Embedding, self).__init__()
        self.d = d

    def forward(self, x):
        """
        
        :param x: N x d
        :return: N x output_dim
        """
        raise NotImplementedError