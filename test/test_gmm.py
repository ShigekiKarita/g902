import numpy

from g902.gmm import GMM


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
