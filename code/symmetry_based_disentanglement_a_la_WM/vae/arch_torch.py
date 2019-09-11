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


	def forward(self, x, encode=False, mean=False, decode=False):
		if decode:
			return self.decode(x)
		mu_and_logvar = self.encode(x)
		z = self.reparameterize(mu_and_logvar)
		if encode:
			if mean:
				mu = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[0]
				return mu
			return z
		return self.decode(z), mu_and_logvar


	def generate_reconstructed_data(self, obs_data, filename):

		images_input = torch.from_numpy(np.array(obs_data)).float().cuda()
		images_output = []
		
		images_output = self.forward(images_input)[0].cpu().detach().numpy()

		images_input = np.array(images_input.cpu().detach()).transpose((0,2,3,1))
		images_output = np.array(images_output).transpose((0,2,3,1))

		out = np.array([images_input, images_output])

		np.save(filename, out)

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
	def __init__(self, path_input):
		self.inputs = []
		for i,elem in enumerate(os.listdir(path_input)):
			try:
				self.inputs.append(np.asarray(Image.open(path_input + elem).convert('RGB')))
			except OSError:
				pass
		self.inputs = np.array(self.inputs).reshape(-1,64,64,3)
		self.inputs = self.inputs.astype('float32') / 255.
		print(self.inputs.shape)
		#self.inputs = np.load('/media/looka/bc6982c9-5dc3-4761-93f3-40a4eafda3ec/phd/flatland/WorldModels/flatland_topview/data/PGMRL_validate_obs_data.npy').reshape(300,64,64,3).transpose((0,3,1,2))
		self.inputs = self.inputs.transpose((0,3,1,2))
		print(self.inputs.shape)

	def __getitem__(self, index):
		input_batch = self.inputs[index]
		return input_batch

	def __len__(self):
		count = len(self.inputs)
		return count # of how many examples(images?) you have

