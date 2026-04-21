import torch


class RiemannianArchetypalMapping(torch.nn.Module):
    def __init__(
        self,
        Omega_manifold,
        archetypes,
        reg_lambda=0.0,
        softmax_temperature=1.0,
        pgd_max_iter=200,
        pgd_tol=1e-6,
        pgd_step=None,
        accelerated=True,
    ):
        super().__init__()
        self.Omega_manifold = Omega_manifold
        self.archetypes = archetypes
        self.r = archetypes.shape[0]

        self.reg_lambda = float(reg_lambda)
        self.softmax_temperature = float(softmax_temperature)
        self.pgd_max_iter = int(pgd_max_iter)
        self.pgd_tol = float(pgd_tol)
        self.accelerated = bool(accelerated)

        with torch.no_grad():
            Omega_archetypes = self.Omega_manifold.phi(self.archetypes).reshape(self.r, -1)  # (r, d)
            self.register_buffer("Omega_archetypes", Omega_archetypes)   # (r, d)

            A = Omega_archetypes.T.contiguous()                          # (d, r)
            self.register_buffer("A", A)

            G = A.T @ A                                                  # (r, r)
            self.register_buffer("G", G)
            self.register_buffer("twoG", 2.0 * G)

            col_norm_sq = (A * A).sum(dim=0)                            # (r,)
            self.register_buffer("col_norm_sq", col_norm_sq)

            if pgd_step is None:
                # L = 2 * ||A^T A||_2 = 2 * lambda_max(G)
                evals = torch.linalg.eigvalsh(G)
                L = 2.0 * torch.clamp(evals[-1], min=1e-12)
                step = 1.0 / L
            else:
                step = float(pgd_step)

            self.register_buffer("pgd_step_tensor", torch.tensor(step, dtype=A.dtype, device=A.device))

    def forward(self, x):
        weights = self.w(x).T
        return self.Omega_manifold.barycentre(self.archetypes, weights=weights)

    @staticmethod
    def _project_simplex(v):
        # v: (..., r)
        u, _ = torch.sort(v, dim=-1, descending=True)
        cssv = torch.cumsum(u, dim=-1) - 1.0
        j = torch.arange(1, v.shape[-1] + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / j > 0
        rho = cond.sum(dim=-1, keepdim=True).clamp(min=1)
        theta = cssv.gather(-1, rho - 1) / rho
        return torch.clamp(v - theta, min=0.0)

    def _distance_softmax_bias(self, Y):
        # Y: (batch, d)
        y_norm_sq = (Y * Y).sum(dim=1, keepdim=True)                    # (batch, 1)
        AtY = Y @ self.A                                                # (batch, r)
        dist_sq = y_norm_sq + self.col_norm_sq.unsqueeze(0) - 2.0 * AtY
        dist_sq = torch.clamp(dist_sq, min=0.0)
        dist = torch.sqrt(dist_sq + 1e-12)
        b = torch.softmax(self.softmax_temperature * dist, dim=-1)
        return b, AtY

    def _objective(self, X, AtY, Y_norm_sq, b):
        # X: (batch, r)
        quad = (X @ self.G * X).sum(dim=-1)
        data = quad - 2.0 * (X * AtY).sum(dim=-1) + Y_norm_sq.squeeze(-1)
        reg = self.reg_lambda * (X * b).sum(dim=-1)
        return data + reg

    def w(self, x):
        Y = self.Omega_manifold.phi(x).reshape(x.shape[0], -1)          # (batch, d)
        batch = Y.shape[0]

        b, AtY = self._distance_softmax_bias(Y)
        y_norm_sq = (Y * Y).sum(dim=1, keepdim=True)

        # good feasible init: simplex projection of bias toward closer archetypes
        X = self._project_simplex(torch.softmax(-self.softmax_temperature * torch.sqrt(
            torch.clamp(y_norm_sq + self.col_norm_sq.unsqueeze(0) - 2.0 * AtY, min=0.0) + 1e-12
        ), dim=-1))

        step = self.pgd_step_tensor
        if self.accelerated:
            Z = X.clone()
            t = torch.ones(batch, 1, device=X.device, dtype=X.dtype)

            for _ in range(self.pgd_max_iter):
                grad = Z @ self.twoG - 2.0 * AtY + self.reg_lambda * b
                X_next = self._project_simplex(Z - step * grad)

                diff = torch.max(torch.abs(X_next - X))
                if diff < self.pgd_tol:
                    X = X_next
                    break

                t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
                Z = X_next + ((t - 1.0) / t_next) * (X_next - X)

                X = X_next
                t = t_next
        else:
            for _ in range(self.pgd_max_iter):
                grad = X @ self.twoG - 2.0 * AtY + self.reg_lambda * b
                X_next = self._project_simplex(X - step * grad)

                diff = torch.max(torch.abs(X_next - X))
                X = X_next
                if diff < self.pgd_tol:
                    break

        return X