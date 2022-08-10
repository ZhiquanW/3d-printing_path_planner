import numpy as np
from typing import List
import matplotlib.pyplot as plt

# res_pos : (N,2)
def pos_real2grid(
    real_pos: np.ndarray, grid_size: float, bottom_left_reference: np.ndarray
) -> np.ndarray:
    assert real_pos.shape[0] == 2, "The dimension of the real_pos must be (2, N)"
    return ((real_pos - bottom_left_reference) / grid_size + 0.5).astype("int32")


def pos_real2grid_sperate(
    real_pos_x: np.array,
    real_pos_y: np.array,
    grid_size: float,
    bottom_left_reference: np.array,
):
    assert (
        real_pos_x.shape == real_pos_y.shape
    ), "The dimension of real_pos_x(y) must be equal"
    return (
        ((real_pos_x - bottom_left_reference[0]) / grid_size + 0.5).astype("int32"),
        ((real_pos_y - bottom_left_reference[1]) / grid_size + 0.5).astype("int32"),
    )


def pos_grid2real(
    grid_pos: np.ndarray, grid_size: float, bottom_left_reference: np.ndarray
):
    return bottom_left_reference + grid_pos * grid_size


def pos_grid2real_sperate(
    grid_pos_x: np.array,
    grid_pos_y: np.array,
    grid_size: float,
    bottom_left_reference: np.array,
):
    return (
        bottom_left_reference[0] + grid_pos_x * grid_size,
        bottom_left_reference[1] + grid_pos_y * grid_size,
    )


def draw_map(*, ax, in_map: np.ndarray, pause: float = 0.001):
    ax.pcolor(in_map, cmap=plt.cm.Blues)
    plt.pause(pause)

def debug_here():
    import pdb
    pdb.set_trace()