from __future__ import division
import gym
import numpy as np
from numpy.random import triangular
import torch
from torch._C import dtype
from torch.autograd import Variable
import os
import psutil
import gc

import train
import train_v2
import buffer
import robosuite as suite
from robosuite.wrappers import GymWrapper

#env = gym.make('BipedalWalker-v2') 'Pendulum-v1'

MAX_EPISODES = 1000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
EPS_PER_SAVE = 25
ENV_STRING = 'robosuite'
TRAINING = True
RENDER_ENV = False
RESET_AGENT = False

if (ENV_STRING == 'FetchPickAndPlace-v1') or (ENV_STRING == 'FetchReach-v1'):
	env = gym.make(ENV_STRING,reward_type='dense')
	S_DIM = env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]
	A_DIM = env.action_space.shape[0]
	A_MAX = env.action_space.high[0]
elif ENV_STRING == 'robosuite':
	env =suite.make(
            "Lift",
            robots="Sawyer",                # use Sawyer robot
            use_camera_obs=False,           # do not use pixel observations
            has_offscreen_renderer=False,   # not needed since not using pixel obs
            has_renderer=RENDER_ENV,        # make sure we can render to the screen
            reward_shaping=True,            # use dense rewards
            control_freq=20,                # control should happen fast enough so that simulation looks smooth
        )
	S_DIM = 3+7
	A_DIM = 8
	A_MAX = 1
else:
	env = gym.make(ENV_STRING)
	S_DIM = env.observation_space.shape[0]
	A_DIM = env.action_space.shape[0]
	A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
if
#offset = trainer.load_models(1000,env_string=ENV_STRING)
offset = 0
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print('EPISODE :- ', _ep)
	tot_reward = 0.0
	for r in range(MAX_STEPS):
		if RENDER_ENV:
			env.render()
		
		if (ENV_STRING == 'FetchPickAndPlace-v1') or (ENV_STRING == 'FetchReach-v1'):
			#state=np.float32(observation['observation'])
			state = np.concatenate((observation['observation'], observation['desired_goal']),dtype=np.float32)
		elif ENV_STRING == 'robosuite':
			joint_c = observation['robot0_joint_pos_cos']
			joint_s = observation['robot0_joint_pos_sin']
			#print('Joint Cosines, ', joint_c)
			#print('Joint Sines, ', joint_s)
			joint_angs = np.arctan2(joint_s,joint_c)
			state = np.concatenate((observation['cube_pos'], joint_angs),dtype=np.float32)
		else:
			state = np.float32(observation)

		if TRAINING == True:
			action = trainer.get_exploration_action(state)
		else:
			action = trainer.get_exploitation_action(state)

		# if _ep%5 == 0:
		# 	# validate every 5th episode
		# 	action = trainer.get_exploitation_action(state)
		# else:
		# 	# get action based on observation, use exploration policy here
		# 	action = trainer.get_exploration_action(state)

		if ENV_STRING == 'robosuite':
			new_observation, reward, done, info = env.step(action)
		else:
			new_observation, reward, done, info = env.step(action)

		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue

		if done:
			new_state = None
		else:
			if (ENV_STRING == 'FetchPickAndPlace-v1') or (ENV_STRING == 'FetchReach-v1'):
				#new_state = np.float32(new_observation['observation'])
				new_state = np.concatenate((new_observation['observation'], new_observation['desired_goal']),dtype=np.float32)
			elif ENV_STRING == 'robosuite':
				joint_c = new_observation['robot0_joint_pos_cos']
				joint_s = new_observation['robot0_joint_pos_sin']
				#print('Joint Cosines, ', joint_c)
				#print('Joint Sines, ', joint_s)
				joint_angs = np.arctan2(joint_s,joint_c)
				new_state = np.concatenate((new_observation['cube_pos'], joint_angs),dtype=np.float32)
			else:
				new_state = np.float32(new_observation)
			# push this exp in ram
			ram.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		if TRAINING == True:
			trainer.optimize()

		tot_reward = tot_reward + reward
		if done:
			break
	print('Total Reward: ', tot_reward)

	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if (_ep%EPS_PER_SAVE== 0) and (TRAINING == True):
		trainer.save_models(_ep+offset,ENV_STRING)

trainer.save_models(MAX_EPISODES+offset,ENV_STRING)
print('Completed episodes')
