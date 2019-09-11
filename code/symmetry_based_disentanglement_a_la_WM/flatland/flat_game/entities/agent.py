# TODO: replace by import of all sensors
from sensors.bump_sensor import BumpSensor
from sensors.proximity_sensor import ProximitySensor
from sensors.rgb_sensor import RgbSensor
from sensors.rgbFog_sensor import RgbFogSensor
from sensors.topview_sensor import TopviewSensor
from sensors.smell_sensor import SmellSensor
from sensors.overview_sensor import OverviewSensor

from entities.entity import Entity

from utils.pygame_util import to_pygame

import numpy as np
import pymunk
import pygame
import math

sensor_classes = {
    "bump": BumpSensor,
    "proximity": ProximitySensor,
    "rgb": RgbSensor,
    "rgbFog": RgbFogSensor,
    "topview": TopviewSensor,
    "smell": SmellSensor,
    "overview": OverviewSensor
}


class Agent(Entity):

    def __init__(self, **kwargs):
        """
        Instantiate an agent with the following parameters
        :param pos: 2d tuple or 'random', initial position of the agent
        :param angle: float or 'random', initial orientation of the agent
        :param sensor: dict with the following fields:
            type: can be 'bump', 'depth', 'image', 'smell', 'overview'
            resolution: int, number of direction in which the agent is looking (doesn't make sense for overview sensor)
            angle: double, view angle of the agent, in radians (doesn't make sense for overview sensor)
            range: range of the sensor (doesn't make sense for bump sensor)
            spread: the spread of the sensor units for one directional sensor, if it is too large, the sensor might not
                    object of smaller size. If it is too small, the simulation will be slower.
            display: whether to display a visual representation of the sensor inpyt in an additional window
        :param measurements: list that can contain 'health', 'items', 'fruits', 'poisons', 'goals',
                'x', 'y', 'theta', 'dead'
        :param actions: list that can contain 'forward', 'backward', 'left', 'right', 'turn_left', 'turn_right'
        :param environment: the environment calling the creation of the agent
        :param living_penalty: int, a default reward at every step
        :param speed: double, used to accentuate or lower the effects of the actions, for instance if set to 2, the
                agent will move 2 times more far when applying the action 'forward'
        """
        super(Agent, self).__init__(**kwargs)

        # Define the sensors
        self.sensors = []
        for sensor_dict in kwargs['sensors']:
            sensor_class = sensor_classes[sensor_dict['typeSensor']]
            sensor_parameters = sensor_dict
            self.sensors.append(sensor_class(**sensor_parameters))

        # TODO: refactor and give name : maxRotationSpeed, maxTranslationSpeed, ...

        # Define the speed
        self.speed = kwargs['speed']

        # Define rotation speed
        self.rotation_speed = kwargs['rotation_speed']

        # Define the radius
        self.radius = kwargs['radius']

        # To make sure the agent doesn't go through entities, 10 is the number of substeps we execute in pymunk for
        # one step in the simulator. Executing 10 actualizations instead of 1 in pymunk doesn't impact too much
        # the fps of the simulator
        assert self.radius > self.speed / 10, "The agent's speed is too large compared to its radius."

        # Test the value of the radius. The diameter of the agent has to be at least the maximum spread of
        # the sensors otherwise other agents might oversee it
        assert self.radius >= 5, "The radius of an agent should be at least 5."

        # Define the possible actions
        self.action_space = kwargs['actions']

        # Initialize the measurements
        self.meas = {}
        self.meas_space = kwargs['measurements']
        for key in self.meas_space:
            if key == 'health' and self.env.mode == 'survival':
                self.meas[key] = 100
            else:
                self.meas[key] = 0

        # Initialize reward and penalty
        self.living_penalty = kwargs['living_penalty'] if 'living_penalty' in kwargs else 0
        self.reward = -self.living_penalty

        # Initial state, the real state will be created by the environment
        self.state = {}
        self.state_space = {}

        # Create the body
        inertia = pymunk.moment_for_circle(10, 0, self.radius, (0, 0))
        body = pymunk.Body(10, inertia)
        c_shape = pymunk.Circle(body, self.radius)
        c_shape.elasticity = 1.0
        body.position = self.x, self.y
        c_shape.collision_type = 0
        body.angle = self.angle
        body.entity = self
        self.env.space.add(body, c_shape)
        self.body = body

        # Normalization
        self.normalize_states = kwargs['normalize_states']
        self.normalize_meas = kwargs['normalize_measurements']
        self.normalize_rewards = kwargs['normalize_rewards']
        self.count = kwargs['count'] if 'count' in kwargs else 0
        horizon = self.env.horizon

        # TBD
        """
        if self.normalize_states:
            if self.count > 0:
                self.states_mean = kwargs['states_mean']
                self.states_var = kwargs['states_var']
            self.states_batch = np.zeros(tuple([horizon] + list(self.sensor.shape(self.env))))
        """

        if self.normalize_meas:
            if self.count > 0:
                self.meas_mean = kwargs['measurements_mean']
                self.meas_var = kwargs['measurements_var']
            self.meas_batch = np.zeros((horizon, len(self.meas.keys())))

        if self.normalize_rewards:
            if self.count > 0:
                self.rewards_mean = kwargs['rewards_mean']
                self.rewards_var = kwargs['rewards_var']
            self.rewards_batch = np.zeros((horizon, 1))

    def update_state(self):
        # self.state_space = {sensor.nameSensor: sensor.shape(self.env) for sensor in self.sensors}
        for sensor in self.sensors:
            self.state_space[sensor.nameSensor] = sensor.get_sensory_input(self.env)

    def update_meas(self, key, value):
        # Method to be called during a step in the environment to update the agent's measurements
        if key in self.meas:
            self.meas[key] += value

    def update_health(self, value, mode):
        # Specific method to handle the 'health' and 'dead' measurements
        self.meas['health'] = self.meas['health'] + value
        if self.meas['health'] <= 0 and mode == 'survival':
            self.set_meas('dead', 1)
            self.reward -= 100

    def set_meas(self, key, value):
        if key in self.meas:
            self.meas[key] = value

    def apply_action(self, actions):
        
        longitudinal_velocity = actions.get('longitudinal_velocity', 0)
        lateral_velocity = actions.get('lateral_velocity', 0)
        angular_velocity = actions.get('angular_velocity', 0)

        vx = longitudinal_velocity*math.cos(self.angle) + lateral_velocity*math.cos(self.angle + 0.5*math.pi)
        vy = longitudinal_velocity*math.sin(self.angle) + lateral_velocity*math.sin(self.angle + 0.5*math.pi)
        self.body.velocity = self.speed * pymunk.Vec2d(vx, vy)
        
        self.body.angular_velocity = angular_velocity * self.rotation_speed

    def get_meas(self):
        if self.normalize_meas:
            self.meas_batch[self.env.t] = np.array(list(self.meas.values()))
            if self.count > 0:
                meas_array = (np.array(list(self.meas.values())) - self.meas_mean) / np.sqrt(self.meas_var)
                return {list(self.meas.keys())[i]: meas_array[i] for i in range(len(self.meas.keys()))}
        return self.meas

    def get_state(self):
        if self.normalize_states:
            # TBD
            return self.state
        return self.state

    def get_reward(self):
        if self.normalize_rewards:
            self.rewards_batch[self.env.t] = self.reward
            if self.count > 0:
                return ((self.reward - self.rewards_mean) / np.sqrt(self.rewards_var))[0]
        return self.reward

    def draw(self):
        """
        Draw the agent on the environment screen
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

        circle_center = self.body.position
        p = to_pygame(circle_center, surface)

        circle_edge = circle_center + pymunk.Vec2d(radius, 0).rotated(self.body.angle)
        p2 = to_pygame(circle_edge, surface)
        line_r = 3
        #pygame.draw.lines(surface, pygame.color.THECOLORS["blue"], False, [p, p2], line_r)

    def get_new_averages(self):
        """
        Computes the total averages with the batch corresponding to this simulation
        :returns a dictionary containing the updated means and variances as well as the new count
        """

        batch_count = self.env.t + 1
        new_count = self.count + batch_count

        result = {'count': new_count}

        if self.normalize_states:
            batch_states_mean = np.mean(self.states_batch, axis=0)
            batch_states_var = np.where(np.var(self.states_batch, axis=0) >= 1e-4, np.var(self.states_batch, axis=0), 1e-4)

            if self.count > 0:
                delta = batch_states_mean - self.states_mean
                new_states_mean = self.states_mean + delta * batch_count / new_count
                result['states_mean'] = new_states_mean

                m_a = self.states_var * self.count
                m_b = batch_states_var * batch_count
                new_states_var = (m_a + m_b + np.square(delta) * self.count * batch_count / new_count) / new_count
                result['states_var'] = new_states_var

            else:
                result['states_mean'] = batch_states_mean
                result['states_var'] = batch_states_var
            
        if self.normalize_meas:
            batch_meas_mean = np.mean(self.meas_batch, axis=0)
            batch_meas_var = np.where(np.var(self.meas_batch, axis=0) >= 1e-4, np.var(self.meas_batch, axis=0), 1e-4)

            if self.count > 0:
                delta = batch_meas_mean - self.meas_mean
                new_meas_mean = self.meas_mean + delta * batch_count / new_count
                result['measurements_mean'] = new_meas_mean

                m_a = self.meas_var * self.count
                m_b = batch_meas_var * batch_count
                new_meas_var = (m_a + m_b + np.square(delta) * self.count * batch_count / new_count) / new_count
                result['measurements_var'] = new_meas_var

            else:
                result['measurements_mean'] = batch_meas_mean
                result['measurements_var'] = batch_meas_var

        if self.normalize_rewards:
            batch_rewards_mean = np.mean(self.rewards_batch, axis=0)
            batch_rewards_var = np.where(np.var(self.rewards_batch, axis=0) >= 1e-4, np.var(self.rewards_batch, axis=0), 1e-4)

            if self.count > 0:
                delta = batch_rewards_mean - self.rewards_mean
                new_rewards_mean = self.rewards_mean + delta * batch_count / new_count
                result['rewards_mean'] = new_rewards_mean

                m_a = self.rewards_var * self.count
                m_b = batch_rewards_var * batch_count
                new_rewards_var = (m_a + m_b + np.square(delta) * self.count * batch_count / new_count) / new_count
                result['rewards_var'] = new_rewards_var

            else:
                result['rewards_mean'] = batch_rewards_mean
                result['rewards_var'] = batch_rewards_var

        return result
