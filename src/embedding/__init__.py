import torch

class Embedding(torch.nn.Module):
    def __init__(self, d):
        super(Embedding, self).__init__()
        self.d = d

    def forward(self, x):
        """
        
        :param x: N x [data_dims]
        :return: N x d
        """
        raise NotImplementedError