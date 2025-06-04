import pytest

np = pytest.importorskip("numpy")

from pucb.single_run import step_function_interpolate


def test_step_function_interpolate_docstring_example():
    x = np.array([1, 3, 7])
    y = np.array([0, 1, 2])
    x_expected = np.array([1, 2, 3, 4, 5, 6, 7])
    y_expected = np.array([0, 0, 1, 1, 1, 1, 2])
    x_out, y_out = step_function_interpolate(x, y)
    assert np.array_equal(x_out, x_expected)
    assert np.array_equal(y_out, y_expected)
