from abc import abstractmethod, ABC
from pygame.color import THECOLORS
from matplotlib import pyplot as plt
import numpy as np
import math


def get_rotated_point(x_1, y_1, x_2, y_2, angle, height):
    # Rotate x_2, y_2 around x_1, y_1 by angle.
    x_change = (x_2 - x_1) * math.cos(angle) + \
               (y_2 - y_1) * math.sin(angle)
    y_change = (y_1 - y_2) * math.cos(angle) - \
               (x_1 - x_2) * math.sin(angle)
    new_x = x_change + x_1
    new_y = height - (y_change + y_1)
    return int(new_x), int(new_y)


class Sensor(ABC):

    def __init__(self, **kwargs):
        self.display = kwargs['display']
        
        # TODO: add tests and default cases
        # TODO: refactor names
        
        # Sensor name and type to access it and compute sensors.
        self.nameSensor =  kwargs['nameSensor'] if 'nameSensor' in kwargs else None
        self.typeSensor =  kwargs['typeSensor'] if 'typeSensor' in kwargs else None
        
        # Field of View of the Sensor
        self.fovResolution = kwargs['fovResolution'] if 'fovResolution' in kwargs else None
        self.fovRange = kwargs['fovRange'] if 'fovRange' in kwargs else None
        self.fovAngle = kwargs['fovAngle'] if 'fovAngle' in kwargs else None
        
        # Anchor of the sensor
        # TODO: for now, attach to body.but should be able to attach to head, etc.
        self.bodyAnchor = kwargs['bodyAnchor'] if 'bodyAnchor' in kwargs else None
        # Relative location (polar) and angle wrt body
        self.d_r = kwargs['d_r'] if 'd_r' in kwargs else None
        self.d_theta = kwargs['d_theta'] if 'd_theta' in kwargs else None
        self.d_relativeOrientation = kwargs['d_relativeOrientation'] if 'd_relativeOrientation' in kwargs else None
        
        if self.display:
            self.figure = None
            self.matrix = None

    
    @abstractmethod
    def get_sensory_input(self, env):
        pass

    @abstractmethod
    def update_display(self, env, array):
        pass


    #TODO: Kiiiiillll meeeee. Reeeemoooove meeee.
    @abstractmethod
    def shape(self, env):
        pass

    def reset(self):
        if self.display:
            plt.close()
