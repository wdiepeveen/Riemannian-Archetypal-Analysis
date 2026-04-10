import torch

from src.dimension_reduction import DimensionReductionSolver

class ArchetypalAnalysisSolver(DimensionReductionSolver):
    """ Base class for archetypal analysis of d-dimensional data X = (X_1, ..., X_n) \in R^{d x n} """
    def __init__(self, d, N) -> None:
        super().__init__(d, N)

        self.F = None
        self.G = None

        self.V = None
        self.labels = None

    @torch.no_grad()
    def fit(self,
        X: torch.Tensor,
        r: int,
        N_max: int = -1, 
        max_iter: int = 500,
        alpha_F_init: float = 1.0,
        alpha_G_init: float = 1.0,
        alpha_min: float = 1e-8,
        alpha_max: float = 1e3,
        backtracking: bool = True,
        bt_shrink: float = 0.5,
        bt_max_steps: int = 20,
        tol: float = 1e-6,
        save_G: bool = True,
    ):
        """
        Solve min_{F,G} ||X - X F G||_F^2
        s.t. columns of F lie in Delta_n, columns of G lie in Delta_r.

        X: (d, n)
        F: (n, r)
        G: (r, n)

        Uses alternating projected gradient with BB step sizes.
        The gradient is first projected to the tangent space of the simplex
        (zero column mean), then a tentative step is taken and projected back
        onto the simplex columnwise.
        """
        assert X.shape[0] == self.d, f"Expected input with {self.d} features, got {X.shape[0]}"
        N = X.shape[1]
        if N_max > 0 and N > N_max:
            idx = torch.randperm(N)[:N_max]
            Y = X[:, idx]

            # compute F for subset of data
            self.fit(Y, r, N_max=-1, max_iter=max_iter, alpha_F_init=alpha_F_init, alpha_G_init=alpha_G_init,
                    alpha_min=alpha_min, alpha_max=alpha_max, backtracking=backtracking, bt_shrink=bt_shrink,
                    bt_max_steps=bt_max_steps, tol=tol, save_G=False)
            
            # compute G for full data
            device = X.device

            # initialize G based on V
            XTV = X.T @ self.V
            G = torch.softmax(XTV.T, dim=0)

            G_prev = None
            gradG_prev = None

            fval_prev = self.objective_given_V(X, self.V, G).item()

            for k in range(max_iter):
                grad_G = self.gradient_G_given_V(X, self.V, G)
                dir_G = self.zero_mean_cols(grad_G)

                alpha_G = self.bb_step(G, dir_G, gradG_prev, G_prev,
                                alpha_init=alpha_G_init,
                                alpha_min=alpha_min,
                                alpha_max=alpha_max)
                
                G_old = G.clone()
                gradG_old = dir_G.clone()

                cand_G = self.proj_simplex_cols(G - alpha_G * dir_G)
                if backtracking:
                    base_val = self.objective_given_V(X, self.V, G).item()
                    a = alpha_G
                    G_new = cand_G
                    for _ in range(bt_max_steps):
                        new_val = self.objective_given_V(X, self.V, G_new).item()
                        if new_val <= base_val:
                            break
                        a *= bt_shrink
                        G_new = self.proj_simplex_cols(G - a * dir_G)
                    G = G_new
                    alpha_G = a
                else:
                    G = cand_G

                fval = self.objective_given_V(X, self.V, G).item()

                rel = abs(fval_prev - fval) / max(1.0, abs(fval_prev))
                if rel < tol:
                    break

                G_prev = G_old
                gradG_prev = gradG_old
                fval_prev = fval

            self.G = G
            self.labels = torch.argmax(G, dim=0)

            print(f"Archetypal Analysis solver finished phase II after {k+1} iterations with objective value {fval:.4f} and relative change {rel:.2f}")
        else:
            device = X.device

            # largest minimal distance initialization
            idx = torch.randint(0, N, (1,), device=device)
            V = X[:, idx]
            for _ in range(1, r):
                dist = torch.cdist(X.T, V.T)
                min_dist, _ = dist.min(dim=1)
                next_idx = torch.argmax(min_dist)
                V = torch.cat([V, X[:, next_idx:next_idx+1]], dim=1)
            
            # initialize F and G based on V
            XTV = X.T @ V
            F = torch.softmax(XTV, dim=0)
            G = torch.softmax(XTV.T, dim=0)

            F_prev = None
            G_prev = None
            gradF_prev = None
            gradG_prev = None

            fval_prev = self.objective(X, F, G).item()

            for k in range(max_iter):
                grad_F = self.gradient_F(X, F, G)
                dir_F = self.zero_mean_cols(grad_F)

                alpha_F = self.bb_step(F, dir_F, gradF_prev, F_prev,
                                alpha_init=alpha_F_init,
                                alpha_min=alpha_min,
                                alpha_max=alpha_max)
                F_old = F.clone()
                gradF_old = dir_F.clone()

                cand_F = self.proj_simplex_cols(F - alpha_F * dir_F)
                if backtracking:
                    base_val = self.objective(X, F, G).item()
                    a = alpha_F
                    F_new = cand_F
                    for _ in range(bt_max_steps):
                        new_val = self.objective(X, F_new, G).item()
                        if new_val <= base_val:
                            break
                        a *= bt_shrink
                        F_new = self.proj_simplex_cols(F - a * dir_F)
                    F = F_new
                    alpha_F = a
                else:
                    F = cand_F

                grad_G = self.gradient_G(X, F, G)
                dir_G = self.zero_mean_cols(grad_G)

                alpha_G = self.bb_step(G, dir_G, gradG_prev, G_prev,
                                alpha_init=alpha_G_init,
                                alpha_min=alpha_min,
                                alpha_max=alpha_max)
                
                G_old = G.clone()
                gradG_old = dir_G.clone()

                cand_G = self.proj_simplex_cols(G - alpha_G * dir_G)
                if backtracking:
                    base_val = self.objective(X, F, G).item()
                    a = alpha_G
                    G_new = cand_G
                    for _ in range(bt_max_steps):
                        new_val = self.objective(X, F, G_new).item()
                        if new_val <= base_val:
                            break
                        a *= bt_shrink
                        G_new = self.proj_simplex_cols(G - a * dir_G)
                    G = G_new
                    alpha_G = a
                else:
                    G = cand_G

                fval = self.objective(X, F, G).item()

                rel = abs(fval_prev - fval) / max(1.0, abs(fval_prev))
                if rel < tol:
                    break

                F_prev, G_prev = F_old, G_old
                gradF_prev, gradG_prev = gradF_old, gradG_old
                fval_prev = fval

            self.F = F
            self.V = X @ F
            if save_G:
                self.G = G
                self.labels = torch.argmax(G, dim=0)
                
                print(f"Archetypal Analysis solver finished after {k+1} iterations with objective value {fval:.4f} and relative change {rel:.2f}")
            else:
                print(f"Archetypal Analysis solver finished phase I after {k+1} iterations with objective value {fval:.4f} and relative change {rel:.2f}")

    @torch.no_grad()
    def proj_simplex_cols(self, V: torch.Tensor, z: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
        """
        Project each column of V onto the probability simplex {x >= 0, sum x = z}.
        V: (m, n)
        returns: (m, n)
        """
        U, _ = torch.sort(V, dim=0, descending=True)
        cssv = torch.cumsum(U, dim=0) - z
        j = torch.arange(1, V.shape[0] + 1, device=V.device, dtype=V.dtype).unsqueeze(1)
        cond = U - cssv / j > 0
        rho = cond.sum(dim=0).clamp(min=1)
        theta = cssv.gather(0, (rho - 1).unsqueeze(0)).squeeze(0) / rho.to(V.dtype)
        W = (V - theta.unsqueeze(0)).clamp_min(0.0)
        s = W.sum(dim=0, keepdim=True).clamp_min(eps)
        return z * W / s

    @torch.no_grad()
    def zero_mean_cols(self, M: torch.Tensor) -> torch.Tensor:
        return M - M.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def objective(self, X: torch.Tensor, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        R = X - X @ F @ G
        return (R * R).sum()
    
    @torch.no_grad()
    def objective_given_V(self, X: torch.Tensor, V: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        R = X - V @ G
        return (R * R).sum()

    @torch.no_grad()
    def gradient_F(self, X: torch.Tensor, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        R = X - X @ F @ G
        grad_F = -2.0 * (X.T @ R @ G.T)
        return grad_F

    @torch.no_grad()
    def gradient_G(self, X: torch.Tensor, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        R = X - X @ F @ G
        grad_G = -2.0 * (F.T @ X.T @ R)
        return grad_G
    
    @torch.no_grad()
    def gradient_G_given_V(self, X: torch.Tensor, V: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        R = X - V @ G
        grad_G = -2.0 * (V.T @ R)
        return grad_G

    @torch.no_grad()
    def bb_step(self, X, grad_X, grad_X_prev=None, X_prev=None,
                alpha_init=1.0, alpha_min=1e-8, alpha_max=1e3):
        if X_prev is None or grad_X_prev is None:
            return alpha_init
        S = X - X_prev
        Ydiff = grad_X - grad_X_prev
        sy = (S * Ydiff).sum()
        ss = (S * S).sum()
        if torch.abs(sy) < 1e-16:
            alpha = torch.tensor(alpha_init, device=X.device, dtype=X.dtype)
        else:
            alpha = ss / sy
        alpha = torch.clamp(alpha, min=alpha_min, max=alpha_max)
        return alpha.item()


    