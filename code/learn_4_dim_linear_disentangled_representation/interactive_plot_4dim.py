# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
from constants import *

import cv2
import numpy as np
import cmath
import math
import torch
from matplotlib import pyplot as plt
import os
def plot_complex_numbers(z_coord_1, z_coord_2):
            plt.clf()
            plt.axis([-2,2,-2,2])
            plt.ion()
            plt.show()

            plt.scatter(z_coord_1.real,z_coord_1.imag)
            plt.scatter(z_coord_2.real,z_coord_2.imag)
            plt.show()
            plt.pause(0.001)


def create_figure_and_sliders(name, state_dim):
    """
    Creating a window for the latent space visualization,
    and another one for the sliders to control it.

    :param name: name of model (str)
    :param state_dim: (int)
    :return:
    """
    # opencv gui setup
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 500, 500)
    cv2.namedWindow('slider for ' + name)
    # add a slider for each component of the latent space
    for i in range(state_dim):
        # the sliders MUST be between 0 and max, so we placed max at 100, and start at 50
        # So that when we substract 50 and divide 10 we get [-5,5] for each component
        cv2.createTrackbar(str(i), 'slider for ' + name, 50, 100, (lambda a: None))


def main():

    all_subdirs = ['vae/' + d for d in os.listdir('vae/') if os.path.isdir('vae/' + d)]

    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    vae = torch.load(latest_subdir + '/saved_models/epoch_34_env_0', map_location={'cuda:0': 'cpu'})

    #vae = torch.load('vae/15498803979771774/saved_models/epoch_34_env_0', map_location={'cuda:0': 'cpu'})

    A_1 = vae.A_1.weight.detach().numpy()
    A_2 = vae.A_2.weight.detach().numpy()
    A_3 = vae.A_3.weight.detach().numpy()
    A_4 = vae.A_4.weight.detach().numpy()

    #import pdb; pdb.set_trace()


    print(A_4)

    #import pdb; pdb.set_trace()

    fig_name = "Decoder for the VAE"

    # TODO: load data to infer bounds
    bound_min = -10
    bound_max = 10

    create_figure_and_sliders(fig_name, 2)

    state = np.array([1,1,1,1])

    should_exit = False
    while not should_exit:

        z_coord_1 = cmath.rect(state[0],state[1])
        z_coord_2 = cmath.rect(state[2],state[3])


        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if k == ord('z'):
            print('up')
            action = [2]
            next_state =  np.dot(A_2, state)
            state = next_state

        if k == ord('q'):
            print('left')
            action = [1]
            next_state = np.dot(A_3, state)
            state = next_state

        if k == ord('d'):
            print('right')
            action = [3]
            next_state = np.dot(A_4, state)
            state = next_state

        if k == ord('s'):
            print('down')
            action = [0]
            next_state = np.dot(A_1, state)
            state = next_state

        print(state)

        #print(A_1)

    


        #plot_complex_numbers(z_coord_1, z_coord_2)

        
        state_polar = np.array([z_coord_1.real,z_coord_1.imag,z_coord_2.real,z_coord_2.imag])
        #state_polar = np.array([z_coord_1.real,z_coord_2.imag,z_coord_2.real,z_coord_1.imag])
        #state_polar = np.array([z_coord_1.real,z_coord_2.real,z_coord_1.imag,z_coord_2.imag])



        reconstructed_image = vae.forward(torch.Tensor(state).reshape(1,4),action = [], decode=True).detach().numpy().reshape(3,64,64).transpose((1,2,0))

        # stop if user closed a window
        if (cv2.getWindowProperty(fig_name, 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_name, 0) < 0):
            should_exit = True
            break
        cv2.imshow(fig_name, reconstructed_image)

    # gracefully close
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()