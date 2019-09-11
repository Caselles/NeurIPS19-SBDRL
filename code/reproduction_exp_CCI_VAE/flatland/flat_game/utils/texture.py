from pygame import Surface, PixelArray, SRCALPHA
from scipy.stats import truncnorm
from abc import abstractmethod
import numpy.random as rand
import numpy as np


class Texture(object):

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, width, height):
        pass


class UniformTexture(Texture):

    def __init__(self, a, b):
        super(UniformTexture, self).__init__()
        self.a = a
        self.b = b

    def generate(self, width, height):
        """
        Generate a pygame Surface with pixels following a uniform density
        :param width: the width of the generated Surface
        :param height: the height of the generated Surface
        :return: the pygame Surface
        """

        surface = Surface((width, height))
        pxarray = PixelArray(surface)

        a = np.array(self.a)
        b = np.array(self.b)
        t = rand.rand(width, height, 3)
        for i in range(width):
            for j in range(height):
                pxarray[i, j] = tuple((a + t[i, j] * (b - a)).astype(int))

        return surface


class NormalTexture(Texture):

    def __init__(self, m, d):
        super(NormalTexture, self).__init__()
        self.m = m
        self.d = d

    def generate(self, width, height):
        """
        Generate a pygame Surface with pixels following a normal density of diagonal covariance matrix.
        :param width: the width of the generated Surface
        :param height: the height of the generated Surface
        :return: the pygame Surface
        """

        surface = Surface((width, height), SRCALPHA)
        pxarray = PixelArray(surface)

        m = np.array(self.m)
        d = np.array(self.d)

        t = np.zeros((width, height, 3))

        for c in range(3):
            a, b = (0 - m[c]) / d[c], (255 - m[c])/d[c]
            tc = truncnorm.rvs(a, b, size=width * height)
            t[:, :, c] = tc.reshape(width, height)

        for i in range(width):
            for j in range(height):
                pxarray[i, j] = tuple((d * t[i, j] + m).astype(int))

        return surface


class ColorTexture(Texture):

    def __init__(self, c):
        super(ColorTexture, self).__init__()
        self.c = c

    def generate(self, width, height):
        surface = Surface((width, height))
        surface.fill(self.c)
        return surface


class StripesTexture(Texture):

    def __init__(self, colors, lengths, angle):
        super(StripesTexture, self).__init__()
        self.colors = colors
        self.lengths = lengths
        self.angle = angle
        assert len(self.colors) == len(self.lengths), "Parameters 'lengths' and 'colors' should be the same length."

    def generate(self, width, height):
        """
        Generate a pygame Surface with pixels following a striped pattern.
        :param width: the width of the generated surface
        :param height: the height of the generated surface
        :return: the pygame Surface
        """

        surface = Surface((width, height), SRCALPHA)
        pxarray = PixelArray(surface)

        for i in range(width):
            for j in range(height):
                l = np.sqrt(i**2 + j**2) * np.cos(np.arctan((j+1)/(i+1)) - self.angle)
                r = l % sum(self.lengths)
                for mode, d in enumerate(np.cumsum(self.lengths)):
                    if r < d:
                        pxarray[i, j] = self.colors[mode]
                        break

        return surface


class PolarStripesTexture(Texture):

    def __init__(self, colors, ratios, iterations):
        super(PolarStripesTexture, self).__init__()
        self.colors = colors
        self.ratios = ratios
        self.iterations = iterations
        assert len(self.colors) == len(self.ratios), "Parameters 'ratios' and 'colors' should be the same length."
        assert sum(self.ratios) == 1, "The color ratios should sum to 1"

    def generate(self, width, height):
        """
        Generate a pyame Surface with pixels following a circular striped pattern from the center of the parent entity
        :param width: the width of the generated surface
        :param height: the height of the generated surface
        :return: the pygame Surface
        """

        surface = Surface((width, height), SRCALPHA)
        pxarray = PixelArray(surface)

        x = width/2
        y = height/2

        for i in range(width):
            for j in range(height):
                if i != x:
                    a = np.arctan((j - y)/(i - x))
                else:
                    a = np.sign(j - y) * np.pi / 2
                r = a % (2 * np.pi / self.iterations)
                for mode, d, in enumerate(np.cumsum(self.ratios)):
                    if r < 2 * np.pi * d / self.iterations:
                        pxarray[i, j] = self.colors[mode]
                        break

        return surface
