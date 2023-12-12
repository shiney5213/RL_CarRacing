from gym.envs.registration import register

register(
    id='CarRacing-v2',
    entry_point='gym_autonmscar.envs:CarRacing-v2Env',
)
