from src.transforms.vector.linear import LinearVectorTransform

class StructuredBlockUpperTriangularVectorTransform(LinearVectorTransform):
    def __init__(self, block_size):
        self.block_size = block_size

    def forward(self, matrix):
        n = matrix.shape[0]
        for i in range(0, n, self.block_size):
            for j in range(0, i, self.block_size):
                matrix[i:i+self.block_size, j:j+self.block_size] = 0
        return matrix