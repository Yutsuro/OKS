import pytest

import numpy as np

from oks import OKS, edit_keypoints


def test_edit_keypoints_basic_flat_list():
    kpts = [10, 20, 1, 30, 40, 0, 50, 60, 2]
    coords, vi = edit_keypoints(kpts)
    assert coords.shape == (3, 2)
    assert vi.shape == (3,)
    np.testing.assert_array_equal(coords, np.array([[10, 20], [30, 40], [50, 60]]))
    np.testing.assert_array_equal(vi, np.array([1, 0, 2]))


def test_edit_keypoints_accepts_numpy_array():
    arr = np.array([[1, 2, 0], [3, 4, 2], [5, 6, 1]])
    coords, vi = edit_keypoints(arr)
    np.testing.assert_array_equal(coords, np.array([[1, 2], [3, 4], [5, 6]]))
    np.testing.assert_array_equal(vi, np.array([0, 2, 1]))


def test_edit_keypoints_raises_on_bad_length():
    bad = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        edit_keypoints(bad)


def test_oks():
    sigma = (
        np.array(
            [
                0.26,
                0.25,
                0.25,
                0.35,
                0.35,
                0.79,
                0.79,
                0.72,
                0.72,
                0.62,
                0.62,
                1.07,
                1.07,
                0.87,
                0.87,
                0.89,
                0.89,
            ]
        )
        / 10.0
    )
    area = 20000
    kpts1 = [
        541,
        299,
        2,
        576,
        280,
        2,
        579,
        315,
        2,
        583,
        351,
        2,
        564,
        394,
        2,
        586,
        316,
        2,
        621,
        351,
        2,
        605,
        406,
        2,
        680,
        320,
        2,
        692,
        310,
        2,
        648,
        322,
        2,
        664,
        391,
        2,
        697,
        316,
        2,
        641,
        356,
        2,
        692,
        419,
        2,
        0,
        0,
        0,
        541,
        287,
        2,
    ]
    kpts2 = [
        542,
        297,
        2,
        584,
        278,
        2,
        574,
        305,
        2,
        586,
        347,
        2,
        565,
        391,
        2,
        584,
        309,
        2,
        620,
        353,
        2,
        612,
        406,
        2,
        696,
        305,
        2,
        685,
        317,
        2,
        0,
        0,
        0,
        659,
        391,
        2,
        699,
        323,
        2,
        644,
        350,
        2,
        689,
        418,
        2,
        0,
        0,
        0,
        542,
        287,
        2,
    ]

    val = OKS(kpts1, kpts2, sigma, area)
    assert val == 0.852633


def test_oks_raises_on_shape_mismatch():
    kpts1 = [0, 0, 1, 10, 10, 1]
    kpts2 = [0, 0, 1, 10, 10, 1, 20, 20, 1]
    sigma = 0.5
    area = 1000

    with pytest.raises(ValueError) as e:
        OKS(kpts1, kpts2, sigma, area)
    assert str(e.value) == "keypoints shape mismatch: (2, 2) vs (3, 2)"


def test_oks_returns_zero_when_all_invisible():
    kpts1 = [10, 20, 0, 30, 40, 1, 50, 60, 0]
    kpts2 = [12, 22, 0, 28, 38, 0, 55, 65, 1]
    sigma = 0.5
    area = 1000

    val = OKS(kpts1, kpts2, sigma, area)

    assert val == 0.0
    assert isinstance(val, float)
