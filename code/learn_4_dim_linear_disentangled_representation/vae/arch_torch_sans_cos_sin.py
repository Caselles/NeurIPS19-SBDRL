import torch.nn as nn
import torch
import torch.nn.functional as F
from constants import *
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image



class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		# Encoder layers
		
		self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(32, 32, 4, stride=2, padding=1)


		self.fc1 = nn.Linear(512, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 2*Z_DIM)

		# Decoder layers

		self.fc4 = nn.Linear(Z_DIM, 256)
		self.fc5 = nn.Linear(256, 512)

		self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.deconv2 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.deconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
		self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

		# Physics layers

		self.A_1 = nn.Linear(Z_DIM, Z_DIM)
		self.A_2 = nn.Linear(Z_DIM, Z_DIM)
		self.A_3 = nn.Linear(Z_DIM, Z_DIM)
		self.A_4 = nn.Linear(Z_DIM, Z_DIM)
		
		
		self.A_1.weight = torch.nn.Parameter(torch.Tensor([[1,1,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]])) 
		self.A_2.weight = torch.nn.Parameter(torch.Tensor([[1,1,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]]))  
		self.A_3.weight = torch.nn.Parameter(torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]]))  
		self.A_4.weight = torch.nn.Parameter(torch.Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]]))
		self.A_1.bias = torch.nn.Parameter(torch.Tensor([0,0,0,0])) 
		self.A_2.bias = torch.nn.Parameter(torch.Tensor([0,0,0,0]))
		self.A_3.bias = torch.nn.Parameter(torch.Tensor([0,0,0,0]))
		self.A_4.bias = torch.nn.Parameter(torch.Tensor([0,0,0,0])) 	


		#import pdb 
		#pdb.set_trace()


		
	def encode(self, x):
		h = F.selu(self.conv1(x))
		h = F.selu(self.conv2(h))
		h = F.selu(self.conv3(h))
		h = F.selu(self.conv4(h))
		self.h_shape = h.shape
		h = h.view(-1, h.shape[1]*h.shape[2]*h.shape[3])
		h = F.selu(self.fc1(h))
		h = F.selu(self.fc2(h))
		h = F.selu(self.fc3(h))
		return h

	def reparameterize(self, mu_and_logvar):
		mu = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[0]
		logvar = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[1]
		std = torch.exp(logvar)
		eps = torch.randn_like(std)
		return eps * std + mu


	def decode(self, z):
		h = F.selu(self.fc4(z))
		h = F.selu(self.fc5(h).reshape(-1, self.h_shape[1], self.h_shape[2], self.h_shape[3]))
		h = F.selu(self.deconv1(h))
		h = F.selu(self.deconv2(h))
		h = F.selu(self.deconv3(h))
		h = F.sigmoid(self.deconv4(h))
		return h

	def predict_next_z(self, z, action, cuda=True):

		if cuda:
			res = torch.Tensor([]).cuda()
		else:
			res = torch.Tensor([])

		for i,ac in enumerate(action):

			if ac == 0:
				z_plus_1 = self.A_1(z[i])

			if ac == 2:
				z_plus_1 = self.A_2(z[i])

			if ac == 1:
				z_plus_1 = self.A_3(z[i])

			if ac == 3:
				z_plus_1 = self.A_4(z[i])

			res = torch.cat((res,z_plus_1.reshape(1,Z_DIM)), dim=0).reshape(-1,Z_DIM)


		return res


	def forward(self, x, action, encode=False, mean=False, decode=False):
		if decode:
			return self.decode(x)
		mu_and_logvar = self.encode(x)
		z = self.reparameterize(mu_and_logvar)
		z_plus_1 = self.predict_next_z(z,action)
		if encode:
			if mean:
				mu = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[0]
				return mu
			return z, z_plus_1
		return self.decode(z), mu_and_logvar, z_plus_1, z


	def generate_reconstructed_data(self, obs_data, actions, filename):

		images_input = torch.from_numpy(np.array(obs_data)).float().cuda()
		images_output = []

		res = self.forward(images_input, actions)

		images_output = res[0].cpu().detach().numpy()

		images_input = np.array(images_input.cpu().detach()).transpose((0,2,3,1))
		images_output = np.array(images_output).transpose((0,2,3,1))

		out = np.array([images_input, images_output])

		np.save(filename, out)

		out_predictions = np.array([res[2].cpu().detach().numpy(), res[3].cpu().detach().numpy()])

		np.save(filename+'_prediction_forward', out_predictions)

		return 

	def linear_interpolation(self, image_origin, image_destination, number_frames):

		res = []
		res.append(image_origin.reshape(1,3,64,64))

		origin_z = self.forward(np.array(image_origin).reshape((1,3,64,64)), encode=True)
		final_z = self.forward(np.array(image_destination).reshape((1,3,64,64)), encode=True)


		for i in range(0, number_frames+1):
			i /= number_frames
			print(i)
			translat_img = ((1 - i) * origin_z) + (i * final_z)
			res.append(self.forward(np.array(translat_img), decode=True))

		res.append(image_destination.reshape(1,3,64,64))

		return np.array(res)

	def generate_rnn_data(self, obs_data, action_data):

		rnn_input = []
		rnn_output = []

		for i, j in zip(obs_data, action_data):    
			rnn_z_input = self.forward(torch.tensor(np.array(i).transpose((0,3,1,2))).cuda(), encode=True).detach().cpu().numpy()
			conc = [np.append(x,y) for x, y in zip(rnn_z_input, j.reshape((300,1)))]
			rnn_input.append(conc[:-1])
			rnn_output.append(np.array(rnn_z_input[1:]))

		rnn_input = np.array(rnn_input)
		rnn_output = np.array(rnn_output)

		return (rnn_input, rnn_output)


class CustomDataset(Dataset):
	def __init__(self, path_input, path_action):

		self.inputs = np.load(path_input).reshape(-1,64,64,3).transpose((0,3,1,2))

		self.actions = np.load(path_action)

		self.actions = np.array(self.actions).reshape(-1,1)



	def __getitem__(self, index):
		input_batch = self.inputs[index]
		action_batch = self.actions[index]
		return input_batch, action_batch

	def __len__(self):
		count = len(self.inputs)
		return count # of how many examples(images?) you have

