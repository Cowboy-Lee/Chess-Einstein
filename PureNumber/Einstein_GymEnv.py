import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from PureNumber.Einstein_PureNumber_InverseStep import GameState_InverseStep as game

PLAYER_RED = 1
PLAYER_BLUE = -1
HORIZONTAL = 0
VERTICAL = 1
ACTIONS = 6

class EinsteinEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    @staticmethod
    def create_instance():
        return EinsteinEnv()

    def __init__(self):
        self.state = game()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, act):
        # 根据行为概率向量act，选择可行行为中的最大者，并将其设置为唯一激活
        action_index = self.state.GetActionIndex(reference_readout=act)
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1
        s_t1, r_t, terminal = self.state.step_in_mind(a_t)
        if terminal:
            ''' 要记得 s_t 里的最后两层应该包含下一次骰子值和下一次的玩家信息 '''
            s_t = self._reset();
        else:
            s_t = s_t1
        return s_t, r_t, terminal, {}

    def _reset(self):
        self.obs, _, _ = self.state.InitializeGame(PLAYER_RED, self.np_random)
        return self.obs

    def _render(self, mode='human', close=False):
        pass

