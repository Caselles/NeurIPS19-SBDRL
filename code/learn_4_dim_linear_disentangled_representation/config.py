import numpy as np
import random
from PIL import Image
from torch import optim
from constants import *
import matplotlib.pyplot as plt
import os
import glob

def adjust_obs(obs):
    obs = np.array(obs)
    return obs.astype('float32') / 255.


def create_optimizer(model, lr, param=None):
    """ Create or load a saved optimizer """

    if ADAM:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_REG)
    
    if param:
        opt.load_state_dict(param)
    
    return opt

def aggregate_data():
    
    res_input = []
    res_output = []

    for i in range(9):
        print(i)
        res_input.append(np.load(savefolder + 'rnn_data/rnn_input_torch_' + str(i)+'.npy').astype(np.float32))
        print(np.array(res_input).shape)
        res_output.append(np.load(savefolder + 'rnn_data/rnn_output_torch_' + str(i)+'.npy'))
        print(np.array(res_output).shape)

    np.save(savefolder + 'rnn_data/rnn_inputs_torch', np.array(res_input))
    np.save(savefolder + 'rnn_data/rnn_outputs_torch', np.array(res_output))

    return 


def plot_figure_loss(running_losses_mean, filename):

    #def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

    def flatten(bad, good = []):
        while bad:
            e = bad.pop()
            if isinstance(e, list):
               bad.extend(e)
            else:
               good.append(e)
        return good

    #fig_loss = flatten([x[0] for x in running_losses_mean])
    #fig_loss_reco = flatten([x[1] for x in running_losses_mean])
    #fig_loss_kl = flatten([x[2] for x in running_losses_mean])
    fig_loss = np.array([x[0] for x in running_losses_mean]).flatten()
    fig_loss_reco = np.array([x[1] for x in running_losses_mean]).flatten()
    fig_loss_kl = np.array([x[2] for x in running_losses_mean]).flatten()

    # plot with various axes scales
    plt.figure(1)

    # loss
    plt.subplot(131)
    plt.plot(range(len(fig_loss)), fig_loss)
    plt.title('Loss')
    plt.grid(True)


    # reco
    plt.subplot(132)
    plt.plot(range(len(fig_loss_reco)), fig_loss_reco)
    plt.yscale('log')
    plt.title('Reconstruction')
    plt.grid(True)


    # kl
    plt.subplot(133)
    plt.plot(range(len(fig_loss_kl)), fig_loss_kl)
    plt.title('KL Loss')
    plt.grid(True)

    plt.savefig(filename)

    plt.close()

    return 

def sliding_avg(data, window_size=10, stride=3):

    avg = [ np.mean(data[i:i+window_size]) for i in range(0, len(data), stride)
                   if i+window_size <= len(data) ]

    return np.array(avg)

def early_stopping(running_losses_mean, threshold):

    early_stopping = False

    loss = np.array(running_losses_mean)[:,0]

    avg = sliding_avg(loss)

    if len(avg) > 10:

        improve_avg = avg[:-1] - avg[1:]

        ref = np.mean(improve_avg[:10])

        condition = np.mean(improve_avg[-10:]) / ref

        print('Early stopping condition is:', condition)

        if condition<threshold:
            early_stopping = True

        return early_stopping

    else:

        return early_stopping

def get_last_file(path):

    list_files = glob.glob(path+'*')

    file = max(list_files, key=os.path.getctime)

    return file






