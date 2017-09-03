import numpy

from g902.gmm import GMM


def test_mean_is_max():
    n_dim = 2
    n_mix = 3
    n_batch = 5
    model = GMM(n_dim, n_mix)
    model.mean[:] = numpy.random.randn(n_mix, n_dim)
    max_prob = model.likelihood(model.mean)  # (n_mix)

    xs = numpy.random.randn(n_batch, n_dim)
    prob = model.likelihood(xs)  # (n_batch)
    assert numpy.alltrue(prob[:, None] < max_prob[None, :])
