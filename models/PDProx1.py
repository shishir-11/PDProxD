import numpy as np

class PDProx():
    def __init__(self, data=None, target=None, C=1, lambda_=0.01, iter=1000, eta=1):
        self.data = data
        self.target = target
        self.C = C
        self.lambda_ = lambda_
        self.iter = iter
        self.w = None
        self.alpha = None
        self.eta = eta
        self.beta = 1 / (1 - np.exp(-eta))  # from paper's rescaling
        self.gamma = None  # will be computed from data in train()

    def compute_gamma(self, X):
        """
        Computes gamma = sqrt(1 / 2c), where c = R^2 / n, R = max row norm
        See Equation (13) and Section 3.1 in the paper
        """
        n = X.shape[0]
        row_norms = np.linalg.norm(X, axis=1)
        R = np.max(row_norms)
        c = R**2 / n
        return np.sqrt(1 / (2 * c))

    def project_onto_simplex(self,v, z=1.0):
        """
        Projects vector v onto the simplex {x | x >= 0, sum(x) <= z}
        Implementation of Duchi et al. (2008): Efficient projections onto the l1-ball
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - z))[0][-1]
        theta = (cssv[rho] - z) / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w

    def box_clipping(self, v, lower=0, upper=None):
        """Projects vector v onto box [lower, upper]"""
        if upper is None:
            upper = self.C
        return np.clip(v, lower, upper)

    def prox_l2(self, u, lmbda):
        """Closed-form proximal operator for non-smooth l2 norm"""
        norm_u = np.linalg.norm(u)
        if norm_u <= lmbda:
            return np.zeros_like(u)
        return (1 - lmbda / norm_u) * u

    def update_w(self, w, G_w, gamma, lambda_):
        """
        Step 5 of Algorithm 1: 
        Proximal step solving:
        min_w 0.5 * ||w - (w - gamma * G_w)||^2 + gamma * lambda * ||w||_2
        """
        u = w - gamma * G_w
        return self.prox_l2(u, gamma * lambda_)

    def train(self):
        """
        PDProx-Dual Algorithm (Algorithm 1)
        for non-smooth ℓ2-regularized exponential α-hinge SVM
        """
        X = self.data
        y = self.target
        m, n = X.shape

        # Step 1: Compute step size γ
        self.gamma = self.compute_gamma(X)

        # Step 2: Initialize variables
        w = np.zeros(n)
        alpha = np.zeros(m)
        dual_beta = np.zeros(m)

        w_sum = np.zeros(n)
        alpha_sum = np.zeros(m)

        for t in range(1, self.iter + 1):
            # Step 3: Compute exp-hinge margin
            margins = 1 - y * (X @ w)
            exp_terms = np.exp(-self.eta * alpha * margins)

            # ∇w (Step 4)
            G_w = -self.beta * X.T @ (alpha * y * exp_terms) / m

            # ∇α (Step 4)
            G_alpha = self.beta * self.eta * margins * exp_terms / m

            # Step 5: Primal update (ℓ2 prox step)
            w_new = self.update_w(w, G_w, self.gamma, self.lambda_)

            # Step 6: Dual α update (box projection)
            alpha_new = self.box_clipping(dual_beta + self.gamma * G_alpha)
            # alpha_new = self.project_onto_simplex(dual_beta + self.gamma * G_alpha, z=self.C)

            # Step 7: Update dual β
            dual_beta = self.box_clipping(dual_beta + self.gamma * G_alpha)
            # dual_beta = self.project_onto_simplex(dual_beta + self.gamma * G_alpha, z=self.C)

            # Step 8: Accumulate for averaging
            w_sum += w_new
            alpha_sum += alpha_new

            # Step 9: Move to next iter
            w = w_new
            alpha = alpha_new

        # Step 10: Final output is averaged iterate
        self.w = w_sum / self.iter
        self.alpha = alpha_sum / self.iter

    def predict(self, X):
        """Signs of dot product with weight vector"""
        return np.sign(X @ self.w)

    def weight_sparsity(self, tol=1e-3):
        """Fraction of nearly zero weights (for sparsity analysis)"""
        return np.sum(np.abs(self.w) < tol) / len(self.w)

    def support_vector_ratio(self, tol=1e-3):
        return np.sum(self.alpha > tol) / len(self.alpha)
