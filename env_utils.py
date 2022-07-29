import numpy as np
from typing import List
import matplotlib.pyplot as plt


def pos_real2grid(
    real_pos: np.ndarray, grid_size: float, bottom_left_reference: np.ndarray
) -> np.ndarray:
    return ((real_pos + 0.5) / grid_size - bottom_left_reference).astype("int32")


def pos_grid2real(
    grid_pos: np.ndarray, grid_size: float, bottom_left_reference: np.ndarray
):
    return bottom_left_reference + grid_pos * grid_size


def draw_map(*, ax, in_map: np.ndarray):
    ax.pcolor(in_map, cmap=plt.cm.Blues)
    plt.pause(0.001)
