from constants import *
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from vae.arch_torch import VAE, CustomDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import time
import torch.nn.functional as F
import numpy as np
import argparse
import os
from config import plot_figure_loss, early_stopping
from PIL import Image


def parse_args():
	desc = "Train VAE on one given maze environment"
	parser = argparse.ArgumentParser(description=desc)
	return parser.parse_args()


def loss_fn(recon_x, x, mu, logvar, anneal):

	batch_size = x.size()[0]
	loss = F.mse_loss(recon_x, x)
	loss = torch.mean(loss)


	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

	return (LAMBDA_RECONSTRUCTION * loss) + (anneal * BETA * kld), LAMBDA_RECONSTRUCTION*loss, BETA*kld


def train_batch(vae, optimizer, frames, anneal):
	""" Train the VAE over a batch of example frames """

	optimizer.zero_grad()
	recon_x, mu_and_logvar = vae(frames)
	mu = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[0]
	logvar = torch.split(mu_and_logvar,int(mu_and_logvar.shape[1]/2),dim=1)[1]
	loss, loss_recon, loss_kl = loss_fn(recon_x, frames, mu, logvar, anneal)

	loss.backward()
	optimizer.step()

	return float(loss), float(loss_recon), float(loss_kl)


def train_vae(folder):
	"""
	Train the VAE.
	"""

	total_ite = 0
	anneal = 1
	env = 0

	## Load or create models
	vae = VAE().to(DEVICE)

	# Create optimizer
	optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

	## Load the dataset
	custom_dataset = CustomDataset('images/')
	# Define data loader
	dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE_VAE)

	# Tensorboard for monitoring
	writer = SummaryWriter(folder)
	fig_loss = []
	fig_loss_reco = []
	fig_loss_kl = []
	running_losses_mean = []
	early_stop = False

	for epoch in range(NUMBER_OF_EPOCHS_VAE):
		running_loss = []
		running_losses = []

		for batch_idx, inputs in tqdm(enumerate(dataset_loader)):

			if batch_idx>2:
				pass

			##############################

			''' TRAINING '''

			inputs = inputs.float().cuda()

			# train and collect loss
			if epoch<0:
				anneal=1
			else:
				anneal = anneal*ANNEAL

			loss, loss_recon, loss_kl = train_batch(vae, optimizer, inputs, anneal)
			running_loss.append(loss)
			running_losses.append([loss, loss_recon, loss_kl])

			##############################

			''' MONITORING & FIGURES '''

			# Monitoring for tensorboard

			if batch_idx % LOG_PERIOD == 0:

				# Log training loss.
				writer.add_scalar('loss/batch_loss', loss, batch_idx) 
				writer.add_scalar('loss/batch_loss_recon', loss_recon, batch_idx) 
				writer.add_scalar('loss/batch_loss_kl', loss_kl, batch_idx) 

			# Figures

			if (batch_idx+1)% FIG_EARLY_STOP_PERIOD ==0:
				print('Anneal value:', anneal)
				running_losses_mean.append(np.mean(running_losses, axis=0))
				print('Loss: ',np.mean(running_losses, axis=0))
				#filename = folder + 'validation/losses_'+str(epoch)+'_env_'+str(env)+'.png'
				filename_n = folder + 'validation/normalized_losses_'+str(epoch)+'_env_'+str(env)+'.png'
				#plot_figure_loss(running_losses_mean, filename)
				plot_figure_loss(running_losses_mean/np.max(running_losses_mean, axis=0), filename_n)

			##############################


			##############################

			''' EARLY STOPPING '''

			if (batch_idx+1)% FIG_EARLY_STOP_PERIOD ==0:

				early_stop = early_stopping(running_losses_mean=running_losses_mean, threshold=THRESHOLD_EARLY_STOPPING)

				if early_stop:
					pass

			##############################


		##############################

		''' MONITORING '''

		if epoch % 1 == 0:

			# Log training loss.
			writer.add_scalar('loss/epoch_loss', np.mean(running_loss), epoch) 

		# Monitor
		print('-------- Epoch', epoch, '/', NUMBER_OF_EPOCHS_VAE, 'finished. Loss on epoch : ', np.mean(running_losses, axis=0))

		##############################

		''' SAVING '''

		if epoch % SAVING_PERIOD == 0:

			torch.save(vae, folder+'saved_models/epoch_'+str(epoch)+'_env_'+str(env))

		##############################

		''' VALIDATING '''

		if epoch % VALIDATE_PERIOD == 0:

			obs_data = []
			for i,elem in enumerate(os.listdir('images/')[:100]):
				obs_data.append(np.asarray(Image.open('images/' + elem).convert('RGB')))
			obs_data = np.array(obs_data).reshape(-1,64,64,3)
			obs_data = obs_data.transpose((0,3,1,2)).astype('float32') / 255.
			filename = folder + 'validation/reconstruction_epoch_'+str(epoch)+'_env_'+str(env)

			vae.generate_reconstructed_data(obs_data=obs_data, filename=filename)

		##############################

		if early_stop:
			pass


	writer.close() # close tensorboard



def main():

	# parse arguments
	args = parse_args()

	if args is None:
		print('Please give arguments!')
	
	folder = 'vae/' + str(time.time()).replace('.', '') + '/' # different folder for any experiment

	if not os.path.exists(folder):
		os.makedirs(folder)
		os.makedirs(folder+'saved_models')
		os.makedirs(folder+'validation')
		print('-------- Vizualize training with: tensorboard --port='+str(8006)+' --logdir='+folder+ ' runs')

	train_vae(folder) # Launch training


if __name__ == "__main__":
	main()
