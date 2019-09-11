from entities.entity import Entity

from utils.pygame_util import to_pygame

import pymunk
import pygame
import math


class Obstacle(Entity):

    def __init__(self, **kwargs):
        """
        Instantiate an obstacle with the following parameters
        :param pos: 2d tuple or 'random', position of the fruit
        :param environment: the environment calling the creation of the fruit
        """
        super(Obstacle, self).__init__(**kwargs)

        self.shape = kwargs['shape']
        assert self.shape in ['circle', 'rectangle', 'composite'], "Unknown shape " + self.shape

        # Define shape parameters
        if self.shape == 'circle':
            self.radius = kwargs['radius']

            # Test the value of the radius. The diameter of the object has to be at least the maximum spread of
            # the sensors otherwise the agents might oversee the object
            assert self.radius >= 5, "The radius of an obstacle should be at least 5."

        elif self.shape == 'rectangle':
            self.width = kwargs['width']
            self.length = kwargs['length']

            # Test the value of the width and length. The width and length of the object have to be at least the
            # maximum spread of the sensors otherwise the agents might oversee the object
            assert self.width >= 10, "The width of an obstacle should be at least 10."
            assert self.length >= 10, "The length of an obstacle should be at least 10."

        elif self.shape == 'composite':
            self.obstacles = []

            for obstacle_parameters in kwargs['obstacles']:
                obstacle_parameters = obstacle_parameters.copy()
                obstacle_parameters['environment'] = self.env
                obstacle_parameters['position'] = (
                    self.x + obstacle_parameters['position'][0],
                    self.y + obstacle_parameters['position'][1]
                )
                obstacle_parameters['angle'] += self.angle
                self.obstacles.append(Obstacle(**obstacle_parameters))

        # Create shape, the body is static
        if self.shape == 'circle':
            shape = pymunk.Circle(
                self.env.space.static_body,
                self.radius,
                offset=(self.x, self.y)
            )
            shape.friction = 1.
            shape.group = 1
            shape.collision_type = 1
            self.env.space.add(shape)

        elif self.shape == 'rectangle':
            shape = pymunk.Poly(
                self.env.space.static_body,
                [
                    (
                        self.x + .5 * (self.length * math.cos(self.angle) - self.width * math.sin(self.angle)),
                        self.y + .5 * (self.length * math.sin(self.angle) + self.width * math.cos(self.angle)),
                    ),
                    (
                        self.x + .5 * (self.length * math.cos(self.angle) + self.width * math.sin(self.angle)),
                        self.y + .5 * (self.length * math.sin(self.angle) - self.width * math.cos(self.angle)),
                    ),
                    (
                        self.x - .5 * (self.length * math.cos(self.angle) - self.width * math.sin(self.angle)),
                        self.y - .5 * (self.length * math.sin(self.angle) + self.width * math.cos(self.angle)),
                    ),
                    (
                        self.x - .5 * (self.length * math.cos(self.angle) + self.width * math.sin(self.angle)),
                        self.y - .5 * (self.length * math.sin(self.angle) - self.width * math.cos(self.angle)),
                    ),
                ]
            )
            shape.friction = 1.
            shape.group = 1
            shape.collision_type = 1
            self.env.space.add(shape)

    def draw(self):
        """
        Draw the obstacle on the environment screen
        """

        if self.shape == 'circle':
            surface = self.env.screen
            radius = int(self.radius)

            # Create a texture surface with the right dimensions
            if self.texture_surface is None:
                self.texture_surface = self.texture.generate(radius * 2, radius * 2)

            # Create the mask
            mask = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            mask.fill((0, 0, 0, 0))
            pygame.draw.circle(mask, (255, 255, 255, 255), (radius, radius), radius)

            # Apply texture on mask
            mask.blit(self.texture_surface, (0, 0), None, pygame.BLEND_MULT)
            mask_rect = mask.get_rect()
            mask_rect.center = to_pygame((self.x, self.y), surface)

            # Blit the masked texture on the screen
            surface.blit(mask, mask_rect, None)

        elif self.shape == 'rectangle':
            surface = self.env.screen
            width = int(self.width)
            length = int(self.length)

            # Create a texture surface with the right dimensions
            if self.texture_surface is None:
                self.texture_surface = self.texture.generate(length, width)
                self.texture_surface.set_colorkey((0, 0, 0, 0))

            # Rotate and center the texture
            texture_surface = pygame.transform.rotate(self.texture_surface, self.angle * 180/math.pi)
            texture_surface_rect = texture_surface.get_rect()
            texture_surface_rect.center = to_pygame((self.x, self.y), surface)

            # Blit the masked texture on the screen
            surface.blit(texture_surface, texture_surface_rect, None)

        elif self.shape == 'composite':
            for obstacle in self.obstacles:
                obstacle.draw()
