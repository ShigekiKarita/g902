from typing import Tuple

import numpy

# from g902 import floatX

floatX = numpy.float64
newaxis = numpy.newaxis


class GMMBase:
    def __init__(self, n_dim: int, n_mix: int = 1):
        self.n_dim = n_dim
        self.n_mix = n_mix
        self.mix = numpy.ones(self.n_mix) / self.n_mix
        # self.mean = numpy.zeros((self.n_mix, self.n_dim))
        self.mean = numpy.random.randn(self.n_mix, self.n_dim)
        self._init_cov()

    def _init_cov(self):
        raise NotImplementedError()

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
        x_m = xs[:, newaxis] - self.mean[newaxis, :]  # (n_batch, n_mix, n_dim)
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

    def update(self, xs: numpy.ndarray) -> Tuple[float, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """

        Args:
            xs: (n_batch, n_mix)

        Returns:
            next_mix: (n_mix)
            next_mean: (n_mix, n_dim)
            next_cov: (n_mix, n_dim, n_dim)
        """
        n_batch, n_dim = xs.shape
        assert n_dim == self.n_dim

        numer = self.components(xs) * self.mix[newaxis]  # (n_batch, n_mix)
        prob_b_m = numer / numer.sum(1)[:, newaxis]  # (n_batch, n_mix)
        log_likelihood = numpy.log(prob_b_m).sum()
        n_population = prob_b_m.sum(0)  # (n_mix)

        next_mix = n_population / n_batch

        next_mean = prob_b_m[:, :, newaxis] * xs[:, newaxis, :]  # (n_batch, n_mix, n_dim)
        next_mean = next_mean.sum(0) / n_population[:, newaxis]  # (n_mix, n_dim)

        x_m = xs[:, newaxis] - next_mean[newaxis, :]  # (n_batch, n_mix, n_dim, 1)
        x_m = x_m.reshape(*x_m.shape, 1)
        next_cov = prob_b_m[:, :, newaxis, newaxis] * (x_m @ x_m.swapaxes(-1, -2))  # (n_batch, n_mix, n_dim, n_dim)
        next_cov = next_cov.sum(0) / n_population[:, newaxis, newaxis]

        assert next_cov.shape == self.cov.shape
        assert next_mix.shape == self.mix.shape
        assert next_mean.shape == self.mean.shape
        return log_likelihood, next_mix, next_mean, next_cov


class GMM(GMMBase):
    def _init_cov(self):
        self.cov = numpy.array([numpy.eye(self.n_dim, dtype=floatX) for _ in range(self.n_mix)])
        # self.cov = numpy.random.randn(self.n_mix, self.n_dim, self.n_dim)

    def fit(self, xs):
        log_likelihood, self.mix, self.mean, self.cov = self.update(xs)
        return log_likelihood


class SphericalGMM(GMMBase):
    def _init_cov(self):
        self._cov_value = numpy.ones(self.n_mix, dtype=floatX)
        self._cov = numpy.array([numpy.eye(self.n_dim) for _ in range(self.n_mix)])

    @property
    def cov(self):
        # numpy.fill_diagonal(self._cov, self._cov_value)
        for c, v in zip(self._cov, self._cov_value):
            numpy.fill_diagonal(c, v)
        return self._cov
