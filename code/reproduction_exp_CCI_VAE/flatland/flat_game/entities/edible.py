from entities.entity import Entity

from utils.pygame_util import to_pygame

import pymunk
import pygame


class Edible(Entity):

    def __init__(self, **kwargs):
        """
        Instantiate an item (fruit or poison) with the following parameters
        :param pos: 2d tuple or 'random', position of the fruit or poison
        :param environment: the environment calling the creation of the fruit or poison
        """
        super(Edible, self).__init__(**kwargs)

        # Define reward
        self.reward = kwargs['reward']
        self.collision_type = kwargs['collision_type']
        self.radius = kwargs['size']

        # Test the value of the radius. The diameter of the object has to be at least the maximum spread of
        # the sensors otherwise the agents might oversee the object
        assert self.radius >= 5, "The size of fruits and poisons should be at least 5."

        # Create body
        inertia = pymunk.moment_for_circle(1, 0, self.radius, (0, 0))
        body = pymunk.Body(1, inertia)
        c_shape = pymunk.Circle(body, self.radius)
        c_shape.elasticity = 1.0
        body.position = self.x, self.y
        c_shape.collision_type = self.collision_type
        body.entity = self
        self.env.space.add(body, c_shape)
        self.body = body

    def draw(self):
        """
        Draw the fruit or poison on the environment screen
        """

        surface = self.env.screen
        shape = list(self.body.shapes)[0]
        radius = int(shape.radius)

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
        mask_rect.center = to_pygame(self.body.position, surface)

        # Blit the masked texture on the screen
        surface.blit(mask, mask_rect, None)
