from sensors.sensor import Sensor
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math


class RgbSensor(Sensor):

    def __init__(self, **kwargs):
        super(RgbSensor, self).__init__(**kwargs)

    def get_sensory_input(self, env):
        
        # Get top view image of environment
        image = env.npimage
        w, h, _ = image.shape
        
        # Position of the agent
        agent_x = int(env.agent.body.position[0])
        agent_y = h - int(env.agent.body.position[1])
        agent_angle = 2*math.pi - env.agent.angle

        # Position and angle of the sensor           
        sensor_x = agent_x + self.d_r * math.cos((agent_angle + self.d_theta) % (2*math.pi))
        sensor_y = agent_y + self.d_r * math.sin((agent_angle + self.d_theta) % (2*math.pi))
        sensor_angle = agent_angle - self.d_relativeOrientation
                    
        polar_img = cv2.linearPolar(image, (sensor_x, sensor_y), self.fovRange, flags=cv2.INTER_NEAREST)
        angle_center = h * (sensor_angle % (2*math.pi))/(2*math.pi)
        rolled_img = np.roll(polar_img, int(h/2 - angle_center), axis=0)
        cropped_img = rolled_img[
                      int(h/2 - h*(self.fovAngle/2.0)/(2*math.pi)):int(h/2+h*(self.fovAngle/2.0)/(2*math.pi)) + 1,
                      :
                      ]
        resized_img = cv2.resize(
            cropped_img,
            (cropped_img.shape[1], int(self.fovResolution)),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Get value sensor
        mask = resized_img != 0
        sensor_id = np.min(np.where(mask.any(axis=1), mask.argmax(axis=1), cropped_img.shape[1] - 1), axis=1)
        
        sensor = resized_img[np.arange(int(self.fovResolution)), sensor_id, :]
        
        if self.display:
            self.update_display(env, sensor)

        env.agent.state[self.nameSensor] = sensor
        
        return sensor

    def update_display(self, env, image):

        height = self.fovResolution * 9 // 16
        width = self.fovResolution

        if self.figure is None:
            self.matrix = np.zeros((height, width, 3))
            plt.figure()
            self.figure = plt.imshow(self.matrix, interpolation=None)
            plt.show(block=False)

        for j in range(height):
            self.matrix[j, :, :] = image[:, :] / 255
        
        self.figure.set_data(self.matrix)
        plt.pause(.0001)
        plt.draw()
        
    def shape(self, env):
        return self.fovResolution, 3
