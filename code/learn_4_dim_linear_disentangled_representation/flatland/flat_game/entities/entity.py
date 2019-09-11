from abc import ABC, abstractmethod
from utils.texture import UniformTexture, NormalTexture, ColorTexture, StripesTexture, PolarStripesTexture
import numpy.random as rand
import math

texture_classes = {
    "random_uniform": UniformTexture,
    "random_normal": NormalTexture,
    "color": ColorTexture,
    "stripes": StripesTexture,
    "polar_stripes": PolarStripesTexture,
}


class Entity(ABC):

    def __init__(self, **kwargs):
        """
        Instantiate an object with the following parameters:
        :param env: Game class, environment instantiating the object
        :param pos: 2d tuple or 'random', initial position of the object
        :param angle: float or 'random', initial orientation of the object
        :param body: pymunk.Body, body of the object in the instantiating environment
        :param texture: Texture class, texture of the shape of the object
        """
        self.env = kwargs['environment']

        # In the simulator we need the object's body position and the object's position to always correspond,
        # to avoid any mistake, x, y, angle and body are defined as properties so that we can easily use setters.

        # Initial body, the real body will be created by the environment
        self.body = None

        # Define initial position and orientation
        if kwargs['position'] == 'random':
            self.__set_x(rand.randint(0, self.env.width))
            self.__set_y(rand.randint(0, self.env.height))
        else:
            self.__set_x(kwargs['position'][0])
            self.__set_y(kwargs['position'][1])

        if 'angle' in kwargs:
            if kwargs['angle'] == 'random':
                self.__set_angle(rand.random() * 2 * math.pi)
            else:
                self.__set_angle(kwargs['angle'])
        else:
            self.__set_angle(0)

        # Define the texture
        if 'texture' not in kwargs:  # Default texture
            kwargs['texture'] = {
                'type': 'color',
                'c': (100, 100, 100)
            }
        texture_class = texture_classes[kwargs['texture']['type']]
        texture_parameters = kwargs['texture'].copy()
        del texture_parameters['type']
        self.texture = texture_class(**texture_parameters)
        self.texture_surface = None

    def __get_x(self):
        if self.body is not None:
            return self.body.position[0]
        return self.__x

    def __set_x(self, x):
        self.__x = x
        if self.body is not None:
            self.body.position[0] = x

    x = property(__get_x, __set_x)

    def __get_y(self):
        if self.body is not None:
            return self.body.position[1]
        return self.__y

    def __set_y(self, y):
        self.__y = y
        if self.body is not None:
            self.body.position[1] = y

    y = property(__get_y, __set_y)

    def __get_angle(self):
        if self.body is not None:
            return self.body.angle
        return self.__angle

    def __set_angle(self, angle):
        self.__angle = angle
        if self.body is not None:
            self.body.angle = angle

    angle = property(__get_angle, __set_angle)

    @abstractmethod
    def draw(self):
        pass
