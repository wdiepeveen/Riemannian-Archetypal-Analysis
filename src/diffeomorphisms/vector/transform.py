from torch.autograd.functional import jvp, vjp

from src.diffeomorphisms.vector import VectorDiffeomorphism
        
class TransformVectorDiffeomorphism(VectorDiffeomorphism):
    """ A vector diffeomorphism implemented using a nflows.transforms object. """
    def __init__(self, d, transform):
        super().__init__(d)

        self.transform = transform

    def forward(self, x):
        """
        Forward pass through the diffeomorphism.
        :param x: N x d
        :return: N x d
        """
        out, _ = self.transform(x, context=None)
        return out

    def inverse(self, y):
        """
        Inverse pass through the diffeomorphism.
        :param y: N x d
        :return: N x d
        """
        out, _ = self.transform.inverse(y, context=None)
        return out
    
    def differential_forward(self, x, X):
        _, jvp_result = jvp(
            lambda x: self.transform(x, context=None)[0],
            (x,),
            (X,),
            create_graph=True,   # <---
            strict=True,
        )
        return jvp_result

    def differential_inverse(self, y, Y):
        _, jvp_result = jvp(
            lambda y: self.transform.inverse(y, context=None)[0],
            (y,),
            (Y,),
            create_graph=True,   # <---
            strict=True,
        )
        return jvp_result

    def adjoint_differential_forward(self, x, X):
        _, vjp_result = vjp(
            lambda x: self.transform(x, context=None)[0],
            (x,),
            (X,),
            create_graph=True,   # <---
            strict=True,
        )
        return vjp_result[0]

    def adjoint_differential_inverse(self, y, Y):
        _, vjp_result = vjp(
            lambda y: self.transform.inverse(y, context=None)[0],
            (y,),
            (Y,),
            create_graph=True,   # <---
            strict=True,
        )
        return vjp_result[0]


    # def differential_forward(self, x, X):
    #     """
    #     Compute the differential map of phi at x for a vector X.
    #     :param x: N x d
    #     :param X: N x d
    #     :return: N x d
    #     """
    #     _, jvp_result = jvp(lambda x: self.transform(x, context=None)[0], (x,), (X,))
    #     return jvp_result

    # def differential_inverse(self, y, Y):
    #     """
    #     Compute the differential map of the inverse of phi at y for a vector Y.
    #     :param y: N x d
    #     :param Y: N x d
    #     :return: N x d
    #     """
    #     _, jvp_result = jvp(lambda y: self.transform.inverse(y, context=None)[0], (y,), (Y,))
    #     return jvp_result
    
    # def adjoint_differential_forward(self, x, X):
    #     """
    #     Compute the adjoint differential map of phi at x for a vector X.
        
    #     :param x: N x d
    #     :param X: N x d
    #     :return: N x d
    #     """
    #     _, vjp_result = vjp(lambda x: self.transform(x, context=None)[0], x, X)
    #     return vjp_result[0]

    # def adjoint_differential_inverse(self, y, Y):
    #     """
    #     Compute the adjoint differential map of the inverse of phi at y for a vector Y.
        
    #     :param y: N x d
    #     :param Y: N x d
    #     :return: N x d
    #     """
    #     _, vjp_result = vjp(lambda y: self.transform.inverse(y, context=None)[0], (y,), (Y,))
    #     return vjp_result[0]
    
