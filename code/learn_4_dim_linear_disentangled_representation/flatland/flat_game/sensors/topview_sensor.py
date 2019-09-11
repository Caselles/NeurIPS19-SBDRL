from sensors.sensor import Sensor
from matplotlib import pyplot as plt
import numpy as np
import numpy as np
from PIL import Image
import pygame
import cv2
import math
import time

class TopviewSensor(Sensor):

    def __init__(self, **kwargs):
        super(TopviewSensor, self).__init__(**kwargs)

    def get_sensory_input(self, env):
        
        # Get top view image of environment
        image = env.npimage
        w, h, _  =  image.shape
        
        # Position of the agent
        agent_x = int( env.agent.body.position[0])
        agent_y = h - int( env.agent.body.position[1])
        agent_angle = 2*math.pi - (env.agent.angle )
        
        
        # Position and angle of the sensor           
        sensor_x =  agent_x + self.d_r * math.cos(( agent_angle  + self.d_theta ) % (2*math.pi))
        sensor_y =  agent_y + self.d_r * math.sin(( agent_angle  + self.d_theta) % (2*math.pi) )
        sensor_angle = agent_angle - self.d_relativeOrientation
        
        angle_topview = sensor_angle #- math.pi/2
                    
        polar_img = cv2.linearPolar(image, (sensor_x, sensor_y), self.fovRange, flags=cv2.INTER_NEAREST+cv2.WARP_FILL_OUTLIERS )
        
        angle_center = h * (( angle_topview ) % (2*math.pi))/ (2*math.pi) 
        rolled_img = np.roll(polar_img, int(h/2 - angle_center), axis = 0 )
        rolled_img[ int(h/2 + h * (self.fovAngle/2.0  )/ (2*math.pi)) + 1:  , : ] = 0
        rolled_img[ : int( h/2 - h * ( self.fovAngle/2.0 )/ (2*math.pi)) , : ] = 0
        
        rolled_img = np.roll(rolled_img, int(h/4), axis = 0 )
        
        reco_img = cv2.linearPolar(rolled_img, ( h/2, h/2 ), self.fovRange, flags=cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS )
        reco_img = reco_img[ int(h/2 - self.fovRange) : int(h/2 + self.fovRange), int(h/2 - self.fovRange) : int(h/2 + self.fovRange), : ]
        
        sensor = reco_img
        
        
        # Get value sensor
       
        if self.display:
            self.update_display(env, sensor)

        #env.agent.state[self.nameSensor] = sensor
        
        
        return sensor

    def update_display(self, env, image):

        height = self.fovRange * 2
        width = self.fovRange * 2

        if self.figure is None:
            self.matrix = np.zeros((height, width, 3))
            plt.figure()
            self.figure = plt.imshow(self.matrix, interpolation=None)
            plt.show(block=False)

        self.matrix[ :] = image[:] / 255
        
        self.figure.set_data(self.matrix)
        plt.pause(.0001)
        plt.draw()
        
    def shape(self, env):
        return self.fovResolution, 3
