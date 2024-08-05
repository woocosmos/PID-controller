import numpy as np
import pandas as pd
import time

class PIDController():
    forward = 16
    turn_left = 15 
    turn_right = 17
    drift_right = 8
    start_line = (-10960, -1288)

    def __init__(self, **kwargs):
        self.dt_ = time.time()

        self.kp = kwargs['kp']
        self.ki = kwargs['ki']
        self.kd = kwargs['kd']

        self.pid_dist = kwargs['dist1']
        self.hair_dist = kwargs['dist2']

        self.left_thresh = kwargs['left']
        self.right_thresh = kwargs['right']

        self.hairpin_index = 0
        self.init_params()

        self.ckpts = pd.read_csv(kwargs['path'])

    def init_params(self): #, x, y
        self.prev_error, self.integral = 0, 0
        self.start_hairpin = False
        
    def set_new_episode(self, x, y, threshold=500):
        if np.sqrt(((self.start_line[0] - x) ** 2 + (self.start_line[1] - y) ** 2)) < threshold:
            self.hairpin_index = 0

    def get_target(self):
        return self.ckpts.iloc[self.hairpin_index].X, self.ckpts.iloc[self.hairpin_index].Y

    def calc(self, dist, dt):

        proportional = dist
        self.integral += dist * dt  # Accumulate error over time
        derivative = (dist - self.prev_error) / dt  # Rate of change of error
        self.prev_error = dist
        
        output = (self.kp * proportional 
                  + self.ki * self.integral 
                  + self.kd * derivative)
        
        # 좌측으로 쏠릴수록 output decrease
        print(output)
        if output > self.left_thresh:
            action = self.turn_left
        elif output < self.right_thresh:
            action = self.turn_right
        else:
            action = self.forward
        return action
    
    def step(self, obs):

        kart_x = obs.position.x
        kart_y = obs.position.y
        is_drift = obs.is_drift

        self.set_new_episode(kart_x, kart_y)
        target_x, target_y = self.get_target()

        nw = time.time()
        dt = nw - self.dt_
        self.dt_ = nw
        
        distance = (
            np.sqrt(((target_x - kart_x) ** 2 + (target_y - kart_y) ** 2))
        )
        
        if self.start_hairpin:
            if distance > self.hair_dist:
                if is_drift:
                    action = self.turn_left
                else:
                    action = self.forward
                    self.hairpin_index += 1
                    self.init_params()
            else:
                action = self.drift_right
        else:
            if distance > self.pid_dist:
                action = self.calc(distance, dt)
            else:
                action = self.drift_right
                self.start_hairpin = True
                
        return action, 12 if is_drift else 9999, [kart_x, kart_y]