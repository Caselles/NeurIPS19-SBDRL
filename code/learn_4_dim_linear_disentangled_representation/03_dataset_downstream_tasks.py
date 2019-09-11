from constants import *
import numpy as np
import torch
import os
from tqdm import tqdm


all_subdirs = ['vae/'+d for d in os.listdir('vae/') if os.path.isdir('vae/'+d)]

latest_subdir = max(all_subdirs, key=os.path.getmtime)

vae = torch.load(latest_subdir+'/saved_models/epoch_10_env_0')


actions = torch.Tensor(np.load('actions.npy')).reshape(-1,1).cuda()

obs = torch.Tensor(np.load('inputs.npy').reshape(-1,64,64,3).transpose((0,3,1,2))).cuda()

states = []

for i,elem in tqdm(enumerate(obs)):

    states.append(vae.forward(elem.reshape(1,3,64,64), action=actions[i], encode=True, mean=True).detach().cpu().numpy())

np.save('../downstream_tasks/states_method_linear', np.array(states).reshape(-1,4))

np.save('../downstream_tasks/actions', actions)