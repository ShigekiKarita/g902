import numpy as np


def test_expand():
    a = np.random.rand(3, 2)
    b = np.random.rand(5, 2)
    c = a[:, np.newaxis] + b[np.newaxis, :]
    assert c.shape == (3, 5, 2)
    for ai in range(len(a)):
        for bi in range(len(b)):
            np.testing.assert_array_almost_equal(c[ai, bi], a[ai] + b[bi])


def test_batch_det():
    a = np.random.randn(2, 2)
    b = np.random.randn(2, 2)
    a_det = np.linalg.det(a)
    b_det = np.linalg.det(b)
    ab = np.array([a, b])
    ab_det = np.linalg.det(ab)
    np.testing.assert_array_almost_equal(ab_det, [a_det, b_det])


def test_batch_inv():
    a = np.random.randn(2, 2)
    a_inv = np.linalg.inv(a)
    b = np.array([a, a_inv])
    b_inv = np.linalg.inv(b)
    np.testing.assert_array_almost_equal(b_inv[0], a_inv)
    np.testing.assert_array_almost_equal(b_inv[1], a)


def test_batch_matmul():
    a = np.random.rand(3, 5, 2)  # (n_batch, n_mix, n_dim)
    b = np.random.rand(5, 2, 2)  # (n_mix, n_dim, n_dim)
    c = a[:, :, np.newaxis] @ b[np.newaxis]
    c = c.squeeze(2)
    assert c.shape == (3, 5, 2)

    for bat_id in range(len(a)):
        for mix_id in range(len(b)):
            np.testing.assert_array_almost_equal(c[bat_id, mix_id], a[bat_id, mix_id] @ b[mix_id])

    d = np.expand_dims(a, -2) @ np.expand_dims(b, 0) @ np.expand_dims(a, -1)
    d = d.squeeze(-2).squeeze(-1)
    assert d.shape == (3, 5)
