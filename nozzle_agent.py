import numpy as np


class NozzleAgent:
    def __init__(
        self,
        *,
        init_pos: list = [0, 0],
        max_speed: float = 1,
        kp: float = 1,
        kd: float = 0.1,
        nozzle_sizze: float = 0.5,
    ) -> None:
        self.pos = np.array(init_pos, dtype=np.float32)
        self.vel = np.zeros(2, dtype=np.float32)
        self.max_speed = max_speed
        self.kp = kp
        self.kd = kd
        self.nozzle_size = nozzle_sizze

    def reset(self,pos = np.zeros(2)):
        self.pos = pos.copy()
        self.vel *= 0
    def drive(
        self, target_vel: np.array, time_step: float = 0.01,
    ):
        assert len(target_vel) == 2
        target_speed = np.linalg.norm(target_vel)
        speed_frac = min(1.0, target_speed / self.max_speed)
        target_vel = speed_frac * target_vel
        vel_offset = target_vel - self.vel
        self.vel += vel_offset * self.kp + self.kd * vel_offset / time_step
        self.pos += self.vel * time_step
        return self.vel, self.pos

    def random_walk(self):
        direction = np.random.uniform(-1, 1, size=2)
        direction /= np.linalg.norm(direction)
        speed = np.random.uniform(0, self.max_speed)
        return speed * direction

