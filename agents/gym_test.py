import gym
import time
# from PureNumber.Einstein_GymEnv import *

# print(gym.envs.registry.all())
# env = gym.make('Go19x19-v0')
# env = EinsteinEnv(draw=True)
env = gym.make('Pendulum-v0')
env.reset()
for _i in range(10):
    env.render()
    act = env.action_space
    done = env.step(act.sample()) # take a random action

    if _i%10==0:
        print(_i)

    time.sleep(1)
