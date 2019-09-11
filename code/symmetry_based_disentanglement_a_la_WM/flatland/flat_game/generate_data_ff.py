from constants import *
import torch
from PIL import Image
import os
import numpy as np

inputs = []
for i, elem in enumerate(os.listdir('images/')):
    im = np.asarray(Image.open('images/' + str(i + 1) + '.png').convert('RGB'))
    inputs.append(im)
inputs = np.array(inputs).reshape(-1, 64, 64, 3)
inputs = inputs.astype('float32') / 255.

all_subdirs = ['vae/'+d for d in os.listdir('vae/') if os.path.isdir('vae/'+d)]

latest_subdir = max(all_subdirs, key=os.path.getmtime)

vae = torch.load(latest_subdir+'/saved_models/epoch_10_env_0', map_location={'cuda:0': 'cpu'})

inputs_z = []
for i, elem in enumerate(inputs):
    z = vae.forward(torch.Tensor(elem.reshape(-1, 64, 64, 3).transpose((0, 3, 1, 2))), encode=True, mean=True)
    inputs_z.append(z.detach().numpy().reshape(1, 2))
inputs_z = np.array(inputs_z).reshape(-1, 2)


np.save('inputs', inputs_z[:-1])
np.save('targets', inputs_z[1:])



