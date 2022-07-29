import env_utils
import matplotlib.pyplot as plt
import numpy as np
import time
import filling_env
import nozzle_agent

if __name__ == "__main__":
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    r = 10
    center = np.array([5, 5])
    agent = nozzle_agent.NozzleAgent(
        init_pos=center, max_speed=2, angular_damping=0.2, kp=10, kd=2, area_size=0.1
    )
    env = filling_env.FillingEnv(
        time_step=0.01, agent=agent, radius=r, center=center, grid_size=0.1
    )
    for _ in range(2):
        env_utils.draw_map(ax=ax, in_map=env.filling_map)
        ax.clear()

    env_utils.draw_map(ax=ax, in_map=env.filling_map)
