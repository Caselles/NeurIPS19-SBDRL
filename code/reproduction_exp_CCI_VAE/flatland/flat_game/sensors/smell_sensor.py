from sensors.sensor import Sensor
from matplotlib import pyplot as plt
import numpy as np


class SmellSensor(Sensor):

    def __init__(self, **kwargs):
        super(SmellSensor, self).__init__(**kwargs)
        self.name = 'smell'

    def get_sensory_input(self, env):
        pass

    def update_display(self, env, image):
        pass
