from sensors.sensor import Sensor
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pygame
import io


class OverviewSensor(Sensor):

    def __init__(self, **kwargs):
        super(OverviewSensor, self).__init__(**kwargs)
        self.name = 'overview'

    def get_sensory_input(self, env):

        height = env.height
        width = env.width
        data = pygame.image.tostring(env.screen, 'RGB')
        pil_image = Image.frombytes('RGB', (width, height), data)
        image = np.asarray(pil_image.convert('RGB'))

        if self.display:
            self.update_display(env, image)

        env.agent.state[self.name] = image
        return image

    def update_display(self, env, image):

        # Since what this sensor sees is the overview of the 2D environment, it be already displayed
        if not env.display:

            height = env.height
            width = env.width

            if self.figure is None:
                self.matrix = np.zeros((height, width, 3))
                plt.figure()
                self.figure = plt.imshow(self.matrix, interpolation=None)
                plt.show(block=False)

            self.matrix = image / 255
            self.figure.set_data(self.matrix)
            plt.draw()

    def shape(self, env):
        return env.width, env.height, 3
