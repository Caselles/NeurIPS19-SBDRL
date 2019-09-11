from sensors.sensor import Sensor
from matplotlib import pyplot as plt
import numpy as np


class BumpSensor(Sensor):

    def __init__(self, **kwargs):
        super(BumpSensor, self).__init__(**kwargs)
        self.range = self.spread
        self.num = 2
        self.name = 'bump'

    def get_sensory_input(self, env):
        self.read(env)
        image = np.zeros(self.shape(env))
        for i in range(self.resolution):
            image[i, 0] = 1 - self.readings[i, 3] / self.range

        if self.display:
            self.update_display(env, image)

        env.agent.state[self.name] = image
        return image

    def update_display(self, env, image):

        height = self.resolution * 9 // 16
        width = self.resolution

        if self.figure is None:
            self.matrix = np.zeros((height, width, 3))
            plt.figure()
            self.figure = plt.imshow(self.matrix, interpolation=None)
            plt.show(block=False)

        for j in range(height):
            for c in range(3):
                self.matrix[j, :, c] = image[:, 0]
        self.figure.set_data(self.matrix)
        plt.draw()

    def shape(self, env):
        return self.resolution, 1
