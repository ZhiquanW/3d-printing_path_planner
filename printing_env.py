from collections import namedtuple
from typing import List
import numpy as np
import nozzle_agent
import env_utils
from bresenham import bresenham
import matplotlib.pyplot as plt
import tqdm
import heartrate
from env_utils import debug_here
import vortex.env.base_env as base_env

# heartrate.trace(browser=True)


class PrintingEnv(base_env.BaseEnv):
    def __init__(
        self,
        *,
        agent: nozzle_agent.NozzleAgent,
        time_step: float,
        step_skip: int,
        eposide_len: int,
        radius: float,
        center: List,
        grid_size: float,
        bottom_left_pos: List,
    ) -> None:
        # store input variables
        self.agent: nozzle_agent.NozzleAgent = agent
        self.time_step: float = time_step
        self.step_skip = step_skip
        self.eposide_len = eposide_len
        self.radius = radius
        self.center = np.array(center, dtype=np.float32)
        self.grid_size: float = grid_size
        self.bottom_left_pos = np.array(bottom_left_pos)
        # input variable legacy check
        self.__bottom_left_reference_check(radius, self.center)
        # compute environment internal data
        self.top_right_pos = 2 * self.center - self.bottom_left_pos
        self.real_world_len = 2 * np.max(np.abs(center - bottom_left_pos))
        self.grid_world_len: int = int(self.real_world_len / grid_size + 0.5)
        self.nozzle_grid_len = int(self.agent.nozzle_size / self.grid_size + 0.5)
        self.grid_world: np.array = self.__init_grid_world()

        # environment internal data
        self.steps = 0
        self.history_pos: np.array = np.zeros((eposide_len, 2))
        self.history_speed: np.array = np.zeros((eposide_len, 1))
        self.history_target_speed: np.array = np.zeros((eposide_len, 1))

        # plot attributes
        self.plot_size = (512, 512)

        self.figure = None
        self.grid_world_ax = None
        self.real_world_ax = None
        self.speed_panel_ax = None

    def __init_grid_world(self,):
        """
            initalize the values in self.grid_world as np.array, a matrix with shape (self.grid_world_len, self.grid_world_len)
        this method should be called during the initaliztaion phase of this class and in self.reset() to reset the self.grid_world to initial values

            1) set grid in target filling area as 0.
            2) set grid outside target filling area as -1.
            3) set filled grid vale += 1 at the filled time step. if a grid is filled multiple times, the value will be higher
        """
        grid_world = (
            np.zeros((self.grid_world_len, self.grid_world_len), dtype=np.int32) - 100
        )
        grid_pos_x, grid_pos_y = np.meshgrid(
            range(self.grid_world_len), range(self.grid_world_len)
        )
        real_pos_x, real_pos_y = env_utils.pos_grid2real_sperate(
            grid_pos_x, grid_pos_y, self.grid_size, self.bottom_left_pos
        )
        r = np.sqrt(
            (real_pos_x - self.center[0]) ** 2 + (real_pos_y - self.center[1]) ** 2
        )
        inside_grid_pos = np.where(r < self.radius)
        grid_world[inside_grid_pos] = 0

        return grid_world

    def __bottom_left_reference_check(self, radius: float, center: np.array):
        corner_pos = center - radius  # in real
        reference_in_real = env_utils.pos_grid2real(
            self.bottom_left_pos, self.grid_size, self.bottom_left_pos
        )
        assert (
            corner_pos[0] > reference_in_real[0] or corner_pos[1] > reference_in_real[1]
        ), f"the setup of the bottom-left reference can not cover all the shape in the map \n bottom_left_reference: {self.bottom_left_pos}, shape corner: {corner_pos}"

    def __pos_outside_check(self, real_pos):
        return np.linalg.norm(real_pos - self.center) > self.radius

    @property
    def obseravtion_dim(self):
        return (self.grid_world_len + 1, max(8, self.grid_world_len))

    @property
    def action_dim(self):
        return 2  # target velocity : (v_x, v_y) in 2d dimension

    def __assembly_observation(self):
        """
        ### this method combines all the necessary information into a numpy 2d array as a state.
        the state is in size of (self.grid_world_len + 1, max(8, self.grid_world_len), and consists of the following elements.
        1. state[:self.grid_world_len, :self.grid_world_len] = grid_world : (self.grid_world_len, self.grid_world_len)
        2. state[self.grid_world_len, :] = [agent position(2), agent velocity(2), max_speed(1), kp(1) ,kd(1), nozzle_size(1)]
        
        * if state_shape[1] == 8:
            state[:self.grid_world_len, self.grid_world_len:] is zero-padding 

        * if state_shape[1] == self.grid_world_len:
            state[self.grid_world_len+1, 8:] is zero-padding
        """
        obs = np.zeros(self.obseravtion_dim)
        obs[: self.grid_world_len, : self.grid_world_len] = self.grid_world
        obs[self.grid_world_len, :2] = self.agent.pos
        obs[self.grid_world_len, 2:4] = self.agent.vel
        obs[self.grid_world_len, 4] = self.agent.max_speed
        obs[self.grid_world_len, 5] = self.agent.kp
        obs[self.grid_world_len, 6] = self.agent.kd
        obs[self.grid_world_len, 7] = self.agent.nozzle_size
        return obs

    def __compute_reward(self, dead: bool, succ: bool):
        if dead:
            return -1000
        if succ:
            return 1000
        filled_r_ratio = 1.0
        filled_grid_num = np.sum(self.grid_world > 0)
        filled_r = filled_grid_num * filled_r_ratio
        time_r_ratio = -0.1
        time_r = 1 * time_r_ratio
        return filled_r + time_r

    def render(self, pause_time: float = 0.01):
        if self.figure == None:
            self.figure = plt.figure(dpi=200)
            grid = plt.GridSpec(3, 4)
            self.grid_world_ax = plt.subplot(grid[:2, :2])
            self.real_world_ax = plt.subplot(grid[:2, 2:])
            self.vel_panel_ax = plt.subplot(grid[2, 0])
        else:
            self.grid_world_ax.clear()
            self.real_world_ax.clear()
            self.vel_panel_ax.clear()
            # draw grid world fillig condition
            # heatmap in matplotlib use x axis as vertical and y axis as horizontal
            self.grid_world_ax.matshow(self.grid_world.T, cmap=plt.cm.Blues)
            self.grid_world_ax.invert_yaxis()
            # self.grid_world_ax.grid(True)

            # draw real world filling condition
            transformed_pos = np.array(self.history_pos[: self.steps]).T
            circle = plt.Circle(self.center, self.radius)
            self.real_world_ax.add_artist(circle)
            self.real_world_ax.plot(
                transformed_pos[0], transformed_pos[1], color="black"
            )

            self.real_world_ax.set_ylim(self.bottom_left_pos[0], self.top_right_pos[0])
            self.real_world_ax.set_xlim(self.bottom_left_pos[1], self.top_right_pos[1])
            # draw velocity panel
            self.vel_panel_ax.plot(
                list(range(len(self.history_speed[: self.steps]))),
                self.history_speed[: self.steps],
                color="green",
            )
            self.vel_panel_ax.plot(
                list(range(len(self.history_target_speed[: self.steps]))),
                self.history_target_speed[: self.steps],
                color="red",
            )
            plt.pause(pause_time)

    def reset(self):
        self.agent.reset(self.center)
        self.grid_world = self.__init_grid_world()
        self.history_pos *= 0
        self.history_speed *= 0
        self.history_target_speed *= 0
        self.steps = 0
        return 0, self.agent.pos, False, {}

    def step(self, action: np.array):
        prev_vel, prev_pos = self.agent.vel.copy(), self.agent.pos.copy()
        for _ in range(self.step_skip):
            next_vel, next_pos = self.agent.drive(action)

        grid_pos = env_utils.pos_real2grid(
            prev_pos, self.grid_size, self.bottom_left_pos
        )
        next_grid_pos = env_utils.pos_real2grid(
            next_pos, self.grid_size, self.bottom_left_pos
        )
        # if the agent position is outside the boundry, then die
        if self.__pos_outside_check(next_pos):
            return (
                self.__compute_reward(True, False),
                self.__assembly_observation(),
                True,
                {},
            )
        # if the self.steps >= self.epodie_len, it means the agent failed to compete the filling task and die
        if self.steps >= self.eposide_len:
            return (
                self.__compute_reward(True, False),
                self.__assembly_observation(),
                True,
                {},
            )
        self.history_pos[self.steps, :] = prev_pos
        self.history_speed[self.steps] = np.linalg.norm(prev_vel)
        self.history_target_speed[self.steps] = np.linalg.norm(action)
        covered_grids_pos = bresenham(
            grid_pos[0], grid_pos[1], next_grid_pos[0], next_grid_pos[1]
        )
        for next_grid_pos in covered_grids_pos:
            grid_range_x = range(
                next_grid_pos[0] - self.nozzle_grid_len - 1,
                next_grid_pos[0] + self.nozzle_grid_len + 1,
            )
            grid_range_y = range(
                next_grid_pos[1] - self.nozzle_grid_len - 1,
                next_grid_pos[1] + self.nozzle_grid_len + 1,
            )
            grid_pos_x, grid_pos_y = np.meshgrid(grid_range_x, grid_range_y)
            real_pos_x, real_pos_y = env_utils.pos_grid2real_sperate(
                grid_pos_x, grid_pos_y, self.grid_size, self.bottom_left_pos
            )
            real_center_pos = env_utils.pos_grid2real(
                np.array(next_grid_pos), self.grid_size, self.bottom_left_pos
            )
            radius = np.sqrt(
                (real_pos_x - real_center_pos[0]) ** 2
                + (real_pos_y - real_center_pos[1]) ** 2
            )
            inside_grid_pos = list(np.where(radius < self.agent.nozzle_size))
            inside_grid_pos[0] += next_grid_pos[0] - self.nozzle_grid_len - 1
            inside_grid_pos[1] += next_grid_pos[1] - self.nozzle_grid_len - 1
            self.grid_world[tuple(inside_grid_pos)] += self.steps

        self.steps += 1
        return (
            self.__compute_reward(False, False),
            self.__assembly_observation(),
            False,
            {},
        )

