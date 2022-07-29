from collections import namedtuple
import numpy as np
import nozzle_agent
import env_utils

FillingMap = namedtuple("FillinggMap", ["map", "filed_area", "total_area"])


class FillingEnv:
    def __init__(
        self,
        *,
        time_step: float,
        agent: nozzle_agent.NozzleAgent,
        radius: float,
        center: np.array,
        grid_size: float,
    ) -> None:
        self.time_step: float = time_step
        self.agent: nozzle_agent.NozzleAgent = agent
        self.grid_size: float = grid_size
        self.map_len: int = int(radius / self.grid_size + 0.5)
        # discretize filling map to pixels, set (0,0) as bottom-left corner
        self.filling_map: np.array = np.zeros(
            (self.map_len, self.map_len), dtype=np.int32
        )
        # bottom left grid pos in real world space
        self.bottom_left_reference = np.zeros(2, dtype=np.int32)
        self.__setup_shape_in_map(radius, center)
        self.__bottom_left_reference_check(radius, center)

    def __setup_shape_in_map(self, radius: float, center: np.array):
        for i in range(self.map_len):
            for j in range(self.map_len):
                real_pos = (
                    self.bottom_left_reference + np.array([i, j])
                ) * self.grid_size
                if np.linalg.norm(real_pos - center) < radius:
                    self.filling_map[i, j] = 1

    def __bottom_left_reference_check(self, radius: float, center: np.array):
        corner_pos = center - radius  # in real
        reference_in_real = env_utils.pos_grid2real(
            self.bottom_left_reference, self.grid_size, self.bottom_left_reference
        )
        assert (
            corner_pos[0] > reference_in_real[0] or corner_pos[1] > reference_in_real[1]
        ), f"the setup of the bottom-left reference can not cover all the shape in the map \n bottom_left_reference: {self.bottom_left_reference}, shape corner: {corner_pos}"

    def reset(self):
        pass

    def step(self, action: np.array):
        _, pos = self.agent.drive(action)
        self.filling_map[env_utils.pos_real2grid(pos)] = 1

