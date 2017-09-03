import numpy


class GMM:
    def __init__(self, n_dim, n_mix=1):
        self.n_dim = n_dim
        self.n_mix = n_mix
        self.mean = numpy.zeros((n_mix, n_dim))
        self.cov = numpy.array([numpy.eye(n_dim) for i in range(n_mix)])
        self.mix = numpy.ones(n_mix) / n_mix

    def components(self, xs: numpy.ndarray) -> numpy.ndarray:
        """
        Args:
            xs: shape is (n_batch, n_dim)

        Returns:
            multiple probability of [N(xs|mean_k, cov_k)$ | k ∈ {1, 2, ..., n_mix}]
            shape is (n_batch, n_mix)
        """
        assert xs.ndim == 2
        assert xs.shape[1] == self.n_dim

        # NOTE: det, inv and matmul(@) support batch-op
        denom = (2 * numpy.pi) ** (self.n_dim / 2) * numpy.linalg.det(self.cov) ** 0.5  # (n_mix)
        x_m = xs[:, numpy.newaxis] - self.mean[numpy.newaxis, :]  # (n_batch, n_mix, n_dim)
        acc = numpy.linalg.inv(self.cov)  # (n_mix, n_dim, n_dim)
        a = numpy.expand_dims(x_m, -2) @ numpy.expand_dims(acc, 0) @ numpy.expand_dims(x_m, -1)
        numer = numpy.exp(-a.squeeze(-2).squeeze(-1))  # (n_batch, n_mix)
        gaussians = numer / denom  # (n_batch, n_mix)
        return gaussians

    def likelihood(self, xs: numpy.ndarray) -> numpy.ndarray:
        """
        Args:
            xs: shape is (n_batch, n_dim)

        Returns:
            normalized probability of $sum_k mix_k N(xs|mean_k, cov_k)$
            shape is (n_batch)
        """
        assert xs.ndim == 2
        assert xs.shape[1] == self.n_dim

        prob = self.components(xs).dot(self.mix)  # '(n_batch)
        return prob
