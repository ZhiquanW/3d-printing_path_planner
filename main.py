import env_utils
import matplotlib.pyplot as plt
import numpy as np
import time
import printing_env
import nozzle_agent

if __name__ == "__main__":
    # import pdb

    # pdb.set_trace()
    r = 4
    center = [5, 5]
    agent = nozzle_agent.NozzleAgent(
        init_pos=center, max_speed=0.5, kp=0.3, kd=0.01, nozzle_sizze=0.5,
    )
    env: printing_env.PrintingEnv = printing_env.PrintingEnv(
        time_step=0.01,
        step_skip=1,
        eposide_len=2000,
        agent=agent,
        radius=r,
        center=center,
        grid_size=0.01,
        bottom_left_pos=np.zeros(2),
    )
    env.render()
    env.reset()
    while True:
        a = agent.random_walk()
        r, s, d, cache = env.step(np.array([1, 2]))
        if d:
            r, s, d, cache = env.reset()
        env.render()
    # import bresenham

    # a = np.arange(100).reshape(10, -1)
    # b = bresenham.bresenham(0, 0, 10, 10)
    # print(a[b])

    # print(tuple(b))
