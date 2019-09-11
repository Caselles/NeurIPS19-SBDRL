# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
from constants import *

import cv2
import numpy as np

import torch
import os

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

    vae = torch.load(latest_subdir + '/saved_models/epoch_10_env_0', map_location={'cuda:0': 'cpu'})

    all_subdirs = ['forward/' + d for d in os.listdir('forward/') if os.path.isdir('forward/' + d)]

    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    ff = torch.load(latest_subdir + '/saved_models/epoch_300_env_0', map_location={'cuda:0': 'cpu'})

    fig_name = "Decoder for the VAE"

    # TODO: load data to infer bounds
    bound_min = -10
    bound_max = 10
    state = np.array([0.,0.])

    create_figure_and_sliders(fig_name, 2)

    should_exit = False
    while not should_exit:
        # stop if escape is pressed
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if k == ord('z'):
            print('up')
            action = [2]
            next_state = ff.forward(torch.Tensor(state).reshape(-1,2),torch.Tensor(action).reshape(-1,1))
            state = next_state

        if k == ord('q'):
            print('left')
            action = [1]
            next_state = ff.forward(torch.Tensor(state).reshape(-1,2),torch.Tensor(action).reshape(-1,1))
            state = next_state

        if k == ord('d'):
            print('right')
            action = [3]
            next_state = ff.forward(torch.Tensor(state).reshape(-1,2),torch.Tensor(action).reshape(-1,1))
            state = next_state

        if k == ord('s'):
            print('down')
            action = [0]
            next_state = ff.forward(torch.Tensor(state).reshape(-1,2),torch.Tensor(action).reshape(-1,1))
            state = next_state



        reconstructed_image = vae.forward(torch.Tensor(state), decode=True).detach().numpy().reshape(3,64,64).transpose((1,2,0))

        # stop if user closed a window
        if (cv2.getWindowProperty(fig_name, 0) < 0) or (cv2.getWindowProperty('slider for ' + fig_name, 0) < 0):
            should_exit = True
            break
        cv2.imshow(fig_name, reconstructed_image)

    # gracefully close
    cv2.destroyAllWindows()

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


if __name__ == '__main__':
    main()