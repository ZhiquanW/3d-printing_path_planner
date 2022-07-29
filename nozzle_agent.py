import numpy as np


class NozzleAgent:
    def __init__(
        self,
        *,
        init_pos: np.array = np.zeros(2),
        max_speed: float = 1,
        angular_damping: float = 0.1,
        kp: float = 10,
        kd: float = 5,
        area_size: float = 0.5,
    ) -> None:
        self.pos = init_pos
        self.vel = np.zeros(3)
        self.max_speed = max_speed
        self.angular_damping = angular_damping
        self.kp = kp
        self.kd = kd
        self.area_size = area_size

    def drive(self, *, time_step: float = 0.01, target_vel: np.array):
        target_vel = max(self.max_speed, target_vel)
        vel_offset = target_vel - self.vel
        self.vel += vel_offset * self.kp + self.kd * vel_offset / time_step
        self.pos += self.vel * time_step
        return self.vel, self.pos
