import torch.nn as nn
import torch
import torch.nn.functional as F
from constants import *
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class FF(nn.Module):
	def __init__(self):
		super(FF, self).__init__()

		# Layers

		self.embedding_action = nn.Embedding(NB_ACTIONS,EMBEDDING_ACTION_SIZE)
		self.fc1 =  nn.Linear(EMBEDDING_ACTION_SIZE+Z_DIM, 256)
		self.fc2 =  nn.Linear(256, 256)
		self.fc3 =  nn.Linear(256, Z_DIM)

		
	def forward(self, z, action):
		#action_embedding = self.embedding_action(action).reshape(-1,1)
		#h = torch.cat((z,action_embedding), dim=1)
		h = torch.cat((z,action.float()), dim=1)
		h = F.selu(self.fc1(h))
		h = F.selu(self.fc2(h))
		z_plus_1 = F.selu(self.fc3(h))
		return z_plus_1

	def validation_forward_model(self, folder, epoch):

		all_subdirs = ['vae/' + d for d in os.listdir('vae/') if os.path.isdir('vae/' + d)]

		latest_subdir = max(all_subdirs, key=os.path.getmtime)

		vae = torch.load(latest_subdir + '/saved_models/epoch_10_env_0', map_location={'cuda:0': 'cpu'})

		if not os.path.exists(folder+'validation/epoch_'+str(epoch)+'up'):

			os.makedirs(folder+'validation/epoch_'+str(epoch)+'/up', exist_ok=True)
			os.makedirs(folder+'validation/epoch_'+str(epoch)+'/down', exist_ok=True)
			os.makedirs(folder+'validation/epoch_'+str(epoch)+'/left', exist_ok=True)
			os.makedirs(folder+'validation/epoch_'+str(epoch)+'/right', exist_ok=True)

		z = torch.tensor([0.,0.])
		im = vae.forward(z,decode=True).detach().numpy().reshape(-1,3,64,64).transpose((0,2,3,1))
		for ac in [-1,0,1,2]:
			for i,action in enumerate(torch.Tensor(np.ones(30))+ac):
				action_str = translate_action(action)
				im = vae.forward(z.cpu(),decode=True).detach().numpy().reshape(-1,3,64,64).transpose((0,2,3,1))
				im = im.reshape(64,64,3) * 255
				im = Image.fromarray(im.astype(np.uint8))
				im.save(folder+'/validation/epoch_'+str(epoch)+'/'+action_str+'/'+str(i)+'.png')
				
				action = action.long()
				next_z = self.forward(z.reshape(-1,2).cuda(),action.reshape(-1,1).cuda())
				z = next_z


def translate_action(action):

	if action == 2:
		res = 'up'
	if action == 0:
		res = 'down'
	if action == 1:
		res = 'left'
	if action == 3:
		res = 'right'

	return res



class CustomDataset(Dataset):
	def __init__(self, path_input, path_action, path_target):
		self.inputs = np.load(path_input)
		self.actions = np.load(path_action)
		self.targets = np.load(path_target)


		self.inputs = np.array(self.inputs).reshape(-1,2)
		print(self.inputs.shape)

		self.actions = np.array(self.actions).reshape(-1,1)
		print(self.actions.shape)

		self.targets = np.array(self.targets).reshape(-1,2)
		print(self.targets.shape)



	def __getitem__(self, index):
		input_batch = self.inputs[index]
		action_batch = self.actions[index]
		target_batch = self.targets[index]
		return input_batch, action_batch, target_batch

	def __len__(self):
		count = len(self.inputs)
		return count # of how many examples(images?) you have

