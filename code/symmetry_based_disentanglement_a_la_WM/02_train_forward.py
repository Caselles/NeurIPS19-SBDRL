from constants import *
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from forward.arch import FF, CustomDataset
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
	desc = "Train FF on one given maze environment"
	parser = argparse.ArgumentParser(description=desc)
	return parser.parse_args()


def loss_fn(prediction, target):

	batch_size = prediction.size()[0]
	loss = F.mse_loss(prediction, target)
	loss = torch.mean(loss)

	return loss


def train_batch(ff, optimizer, inputs, actions, targets, batch_idx):
	""" Train the FF on transitions """

	optimizer.zero_grad()
	predictions = ff(inputs, actions)
	loss = loss_fn(predictions, targets)

	"""
	print('---------------------')

	if batch_idx % 50==0:
		print('save')
		import pickle
		with open('outfile', 'wb') as fp:
			pickle.dump([inputs[:10],actions[:10],predictions[:10],targets[:10]], fp)
	print('---------------------')
	"""


	loss.backward()
	optimizer.step()

	return float(loss)


def train_ff(folder):
	"""
	Train the FF.
	"""

	total_ite = 0
	env = 0

	## Load or create models
	ff = FF().to(DEVICE)

	# Create optimizer
	optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)

	## Load the dataset
	custom_dataset = CustomDataset(path_input='inputs.npy', path_action='actions.npy', path_target='targets.npy')
	# Define data loader
	dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE_FF)

	# Tensorboard for monitoring
	writer = SummaryWriter(folder)
	fig_loss = []
	running_losses_mean = []
	early_stop = False

	for epoch in range(NUMBER_OF_EPOCHS_FF):
		running_loss = []
		running_losses = []

		for batch_idx, (inputs, actions, targets) in tqdm(enumerate(dataset_loader)):

			if batch_idx>2:
				pass

			##############################

			''' TRAINING '''

			inputs = inputs.float().cuda()
			actions = actions.long().cuda()
			targets = targets.float().cuda()

			# train and collect loss

			loss = train_batch(ff, optimizer, inputs, actions, targets, batch_idx)
			running_loss.append(loss)
			running_losses.append([loss])

			##############################

			''' MONITORING & FIGURES '''

			# Monitoring for tensorboard

			if batch_idx % LOG_PERIOD == 0:

				# Log training loss.
				writer.add_scalar('loss/batch_loss', loss, batch_idx) 

			# Figures

			if (batch_idx+1)% FIG_EARLY_STOP_PERIOD ==0:
				running_losses_mean.append(np.mean(running_losses, axis=0))
				print('Loss: ',np.mean(running_losses, axis=0))
				#filename = folder + 'validation/losses_'+str(epoch)+'_env_'+str(env)+'.png'
				#filename_n = folder + 'validation/normalized_losses_'+str(epoch)+'_env_'+str(env)+'.png'
				#plot_figure_loss(running_losses_mean, filename)
				#plot_figure_loss(running_losses_mean/np.max(running_losses_mean, axis=0), filename_n)

			##############################



		##############################

		''' MONITORING '''

		if epoch % 1 == 0:

			# Log training loss.
			writer.add_scalar('loss/epoch_loss', np.mean(running_loss), epoch) 

		# Monitor
		print('-------- Epoch', epoch, '/', NUMBER_OF_EPOCHS_FF, 'finished. Loss on epoch : ', np.mean(running_losses, axis=0))

		##############################

		''' SAVING '''

		if epoch % SAVING_PERIOD == 0:

			torch.save(ff, folder+'saved_models/epoch_'+str(epoch)+'_env_'+str(env))

		##############################

		''' VALIDATING '''


		if epoch % VALIDATE_PERIOD == 0:

			ff.validation_forward_model(folder, epoch)


		##############################

		if early_stop:
			pass


	writer.close() # close tensorboard



def main():

	# parse arguments
	args = parse_args()

	if args is None:
		print('Please give arguments!')
	
	folder = 'forward/' + str(time.time()).replace('.', '') + '/' # different folder for any experiment

	if not os.path.exists(folder):
		os.makedirs(folder)
		os.makedirs(folder+'saved_models')
		os.makedirs(folder+'validation')
		print('-------- Vizualize training with: tensorboard --port='+str(8006)+' --logdir='+folder+ ' runs')

	train_ff(folder) # Launch training


if __name__ == "__main__":
	main()
