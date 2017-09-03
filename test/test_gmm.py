import numpy

from g902.gmm import GMM, SphericalGMM


def test_mean_is_max():
    n_dim = 2
    n_mix = 3
    n_batch = 5

    model = GMM(n_dim, n_mix)
    model.mean[:] = numpy.random.rand(n_mix, n_dim)
    max_prob = model.components(model.mean)  # (n_mix, n_mix)

    xs = numpy.random.rand(n_batch, n_dim)
    batch_prob = model.components(xs)  # (n_batch, n_mix)
    print(model.mean)
    print(xs)
    for m in range(n_mix):
        assert numpy.all(batch_prob[:, m] <= max_prob[m, m])


def test_spherical_cov():
    n_dim = 2
    n_mix = 3
    model = SphericalGMM(n_dim, n_mix)
    model._cov_value = numpy.arange(0, n_mix)
    assert model.cov.shape == (n_mix, n_dim, n_dim)
    for m in range(n_mix):
        for i in range(n_dim):
            for j in range(n_dim):
                v = model.cov[m, i, j]
                if i == j:
                    assert v == m
                else:
                    assert v == 0.0
