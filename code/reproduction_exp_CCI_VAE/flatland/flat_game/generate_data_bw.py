from env import Env
import numpy as np
from tqdm import trange
import time
import math
import pickle as pk
import random
import pymunk

print(pymunk.version)

agent_parameters = {
    'radius': 5,
    'speed': 10,
    'rotation_speed' : math.pi/8,
    'living_penalty': 0,
    'position': 'random',
    'angle': 'random',
    'sensors': [
      
        {
           'nameSensor' : 'proximity_test',
           'typeSensor': 'proximity',
           'fovResolution': 64,
           'fovRange': 300,
           'fovAngle': math.pi ,
           'bodyAnchor': 'body',
           'd_r': 0,
           'd_theta': 0,
           'd_relativeOrientation': 0,
           'display': False,
        }
        
       
    ],
    'actions': ['forward', 'turn_left', 'turn_right', 'left', 'right', 'backward'],
    'measurements': ['health', 'poisons', 'fruits'],
    'texture': {
        'type': 'color',
        'c': (0, 0, 255)
    },
    'normalize_measurements': False,
    'normalize_states': False,
    'normalize_rewards': False
}

env_parameters = {
    'map':False,
    'n_rooms': 2,
    'display': True,
    'horizon': 1,
    'shape': (64, 64),
    'mode': 'time',
    'poisons': {
        'number': 0,
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
        'number': 0,
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
       
    ],
    'walls_texture': {
        'type': 'color',
        'c': (1, 1, 1)
    },
    'agent': agent_parameters
}


env = Env(**env_parameters)
n = len(agent_parameters['actions'])
meas, sens = None, None
prev_sens = None

dataset = []

action = {}

longtrans = 0
lattrans = 0
rot = 0

start = time.time()
done = False
for i in trange(10000):
    while not done:
        if sens is not None:
            prev_sens = sens.copy()
            
        action['longitudinal_velocity'] = random.uniform(0,1)
        action['lateral_velocity'] = random.uniform(-0.2,0.2)
        action['angular_velocity'] = random.uniform(-1,1)
            
        #action = agent_parameters['actions'][np.random.choice(n, 1)[0]]
        sens, r, done, meas = env.step(action)
        
    env.reset()
    done = False
end = time.time()

print(end - start)


from PIL import Image
import os
inputs = []
for i,elem in enumerate(os.listdir('images/')):
    try:
        inputs.append(np.asarray(Image.open('images/' + elem).convert('RGB')))
    except OSError:
        pass
inputs = np.array(inputs).reshape(-1,64,64,3)
inputs = inputs.astype('float32') / 255.

np.save('images/data', inputs)

