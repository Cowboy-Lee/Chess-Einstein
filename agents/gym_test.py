import gym

print(gym.envs.registry.all())
env = gym.make('Go19x19-v0')
env.reset()
for _i in range(1000):
    env.render()
    done = env.step(env.action_space.sample()) # take a random action

    if _i%10==0:
        print(_i)
