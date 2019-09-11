from env import Env
import numpy as np
from pygame.locals import *
import pygame
import time
import math

x = np.random.randint(600, 800)
y = np.random.randint(50, 250)

agent_parameters = {
    'radius': 20,
    'speed': 20,
    'living_penalty': 0,
    'position': (x, y),
    'angle': 'random',
    'sensors': [
        {
           'type': 'rgb_fog',
           'resolution': 64,
           'range': 300,
           'angle': math.pi * 90 / 180,
           'spread': 10,
           'display': True
        },
        {
           'type': 'topview',
           'resolution': 64,
           'range': 300,
           'angle': math.pi * 90 / 180,
           'spread': 10,
           'size_topview': 100,
           'display': False
        }
    ],
    'actions': ['forward', 'turn_left', 'turn_right'],
    'measurements': ['health', 'poisons', 'fruits'],
    'texture': {
        'type': 'color',
        'c': (200, 200, 200)
    },
    'normalize_measurements': False,
    'normalize_states': False,
    'normalize_rewards': False
}

env_parameters = {
    'display': True,
    'horizon': 500,
    'shape': (900, 600),
    'mode': 'time',
    'poisons': {
        'number': 20,
        'positions': 'random',
        'size': 10,
        'reward': -10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (150, 0, 200),
        }
    },
    'fruits': {
        'number': 20,
        'positions': 'random',
        'size': 10,
        'reward': 10,
        'respawn': True,
        'texture': {
            'type': 'color',
            'c': (255, 150, 0),
        }
    },
    'obstacles': [
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 300,
            'angle': 0,
            'position': (150, 150),
            'texture': {
                'type': 'color',
                'c': (100, 0, 0),
            }
        },
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 300,
            'angle': math.pi/2,
            'position': (500, 500),
            'texture': {
                'type': 'color',
                'c': (50, 50, 0),
            }
        },
        {
            'shape': 'rectangle',
            'width': 50,
            'length': 500,
            'angle': 3*math.pi/4,
            'position': (350, 300),
            'texture': {
                'type': 'color',
                'c': (0, 0, 100),
            }
        },
        {
            'shape': 'circle',
            'position': (700, 300),
            'radius': 100,
            'texture': {
                'type': 'color',
                'c': (0, 50, 50),
            }
        }
    ],
    'walls_texture': {
        'type': 'color',
        'c': (50, 50, 50)
    },
    'agent': agent_parameters
}


"""
To play:
z = go forward
left arrow = turn left
right arrow = turn right

For it to work, you still have to have the game display ON, and this should be the active window.
If you want to play in partially observable conditions, just don't look at this display window :)
"""

env = Env(**env_parameters)
n = len(agent_parameters['actions'])
meas, sens = None, None

start = time.time()
done = False
for i in range(5):
    time.sleep(1)
    while not done:
        for event in pygame.event.get():

            if event.type == KEYDOWN:
                if event.key == K_z:
                    sens, r, done, meas = env.step('forward')
                    print(meas)
                    print(r)
                if event.key == K_LEFT:
                    sens, r, done, meas = env.step('turn_left')
                    print(meas)
                    print(r)
                if event.key == K_RIGHT:
                    sens, r, done, meas = env.step('turn_right')
                    print(meas)
                    print(r)
            if done:
                break

    env.reset()
    done = False
end = time.time()

print(end - start)


