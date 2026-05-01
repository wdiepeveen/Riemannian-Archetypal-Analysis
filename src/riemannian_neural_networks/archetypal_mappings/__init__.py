import torch

from src.riemannian_neural_networks import RiemannianNeuralNetwork

class RiemannianArchetypalMapping(RiemannianNeuralNetwork):
    def __init__(
        self,
        manifold,
        archetypes,
        max_iter=500,
        tol=1e-6,
        accelerated=True,
        line_search=False,
        ls_beta=0.5,
        ls_c=1e-4,
        ls_max_iter=20,
        ls_min_step=1e-12,
    ):
        super().__init__(manifold)
        self.m = archetypes
        self.r = archetypes.shape[0]

        self.max_iter = max_iter
        self.tol = tol
        self.accelerated = accelerated
        self.line_search = line_search

        self.ls_beta = ls_beta
        self.ls_c = ls_c
        self.ls_max_iter = ls_max_iter
        self.ls_min_step = ls_min_step

        # Subclasses should set self.step_size.
        # We then cache it as the reference step and maintain a mutable working step.
        # self.step_size = None
        self.initial_step_size = None
        self.current_step_size = None

    def _initialize_step_sizes(self):
        if self.initial_step_size is None:
            raise ValueError(
                "self.initial_step_size must be set by the subclass before calling archetype_weights."
            )
        # if self.initial_step_size is None:
        #     self.initial_step_size = self.step_size
        if self.current_step_size is None:
            self.current_step_size = self.initial_step_size

    def forward(self, x, return_weights=False):
        """
        Computes T(x) := argmin_{y} \sum_{j=1}^r g_j^* d(y, m^j)^2,
        where g^* are the archetype weights implemented by subclass.
        :param x: N x [input_dim] tensor
        """
        weights = self.archetype_weights(x)
        barycentre = self.manifold.barycentre(self.m, weights=weights.T)
        if return_weights:
            return barycentre, weights
        return barycentre

    def archetype_weights(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        self._initialize_step_sizes()

        w = self.archetype_weights_init(x)
        N = x.shape[0]

        if self.accelerated:
            v = w.clone()
            t = torch.ones(N, 1, device=x.device, dtype=x.dtype)

            for _ in range(self.max_iter):
                grad_v = self.gradient(v, x)

                if self.line_search:
                    w_next, accepted_step = self._projected_backtracking(v, grad_v, x)
                    if accepted_step < self.ls_min_step:
                        print(f"Iteration {_} Step size below minimum threshold. Stopping optimization.")
                        w = w_next
                        break
                else:
                    accepted_step = self.current_step_size
                    w_next = self._project_simplex(v - accepted_step * grad_v)

                diff = torch.max(torch.abs(w_next - w))
                if diff < self.tol:
                    print(f"Iteration {_} Convergence achieved with diff {diff:.2e}. Stopping optimization.")
                    w = w_next
                    break

                t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
                v_next = w_next + ((t - 1.0) / t_next) * (w_next - w)

                if self.line_search:
                    f_w_next = self.objective(w_next, x).sum()
                    f_v_next = self.objective(v_next, x).sum()
                    if f_v_next > f_w_next:
                        v_next = w_next
                        t_next = torch.ones_like(t)

                w = w_next
                v = v_next
                t = t_next
                self.current_step_size = accepted_step
        else:
            for _ in range(self.max_iter):
                grad_w = self.gradient(w, x)

                if self.line_search:
                    w_next, accepted_step = self._projected_backtracking(w, grad_w, x)
                    self.current_step_size = accepted_step
                else:
                    accepted_step = self.current_step_size
                    w_next = self._project_simplex(w - accepted_step * grad_w)

                diff = torch.max(torch.abs(w_next - w))
                w = w_next
                if diff < self.tol:
                    print(f"Iteration {_} Convergence achieved with diff {diff:.2e}. Stopping optimization.")
                    break

        return w

    def archetype_weights_init(self, x):
        """
        :param x: N x [input_dim] tensor
        :return: N x r tensor of archetypal coefficients
        """
        raise NotImplementedError("Subclasses must implement archetype_weights_init method")

    def objective(self, w, x):
        """
        Computes the objective function value for the given archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N tensor representing the objective function value
        """
        raise NotImplementedError("Subclasses must implement objective method")

    def gradient(self, w, x):
        """
        Computes the gradient of the objective function with respect to the archetypal coefficients.
        :param w: N x r tensor of archetypal coefficients
        :param x: N x [input_dim] tensor
        :return: N x r tensor of gradients
        """
        raise NotImplementedError("Subclasses must implement gradient method")

    def _projected_backtracking(self, w, grad_w, x):
        """
        Armijo backtracking for projected gradient descent on the simplex.

        Uses the projected candidate
            w_trial = P_{\Delta}(w - alpha * grad_w)
        and accepts alpha when
            f(w_trial) <= f(w) + c * <grad_w, w_trial - w>.

        :param w: N x r tensor
        :param grad_w: N x r tensor
        :param x: N x [input_dim] tensor
        :return: (accepted iterate, accepted step size)
        """
        alpha = self.current_step_size if self.current_step_size is not None else self.initial_step_size
        f_w = self.objective(w, x).sum()

        w_trial = w
        for _ in range(self.ls_max_iter):
            w_trial = self._project_simplex(w - alpha * grad_w)
            f_trial = self.objective(w_trial, x).sum()

            rhs = f_w + self.ls_c * torch.sum(grad_w * (w_trial - w))
            if f_trial <= rhs:
                return w_trial, alpha

            alpha *= self.ls_beta

        return w_trial, alpha

    @staticmethod
    def _project_simplex(v):
        """
        Projects each row of v onto the probability simplex.
        :param v: N x r tensor
        :return: N x r tensor, projection of v onto the probability simplex
        """
        u, _ = torch.sort(v, dim=-1, descending=True)
        cssv = torch.cumsum(u, dim=-1) - 1.0
        j = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / j > 0
        rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        theta = cssv.gather(-1, rho - 1) / rho
        return torch.clamp(v - theta, min=0.0)