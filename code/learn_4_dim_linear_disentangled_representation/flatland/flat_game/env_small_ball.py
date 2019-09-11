import random
import math
import pygame
from pygame.color import THECOLORS

import pymunk

from entities.agent import Agent
from entities.edible import Edible
from entities.obstacle import Obstacle
from PIL import Image

from maps.map import Dungeons

import numpy as np

class Env(object):

    def __init__(self, **kwargs):
        """
        Instantiate a game with the given parameters
        :param horizon: int, time horizon of an episode
        :param done: bool, True if the episode is terminated
        :param mode: 'goal' or 'health':
            - if 'goal' : we use the field goal to create a goal and the simulation ends
                when the goal is reached or when we reach the horizon
            - if 'survival', the health measurements in initialized to 100 and the simulation
                ends when the health reaches 0 or when we reach the horizon
        :param shape: size 2 tuple with height and width of the environment
        :param goal: dict with the following fields, only useful if mode is 'goal'
            - size: float, size of the goal
            - position: size 2 tuple giving the position or 'random'
        :param walls: dict with the following fields:
            - number: int, number of walls in the environment
            - size: float, size of the walls
            - position: array of coordinates or 'random'
        :param poisons: dict with the following fields
            - number: int, number of poisons in the environment
            - size: float, size of the poisons
            - reap: bool, whether another poison object reappears when one is consumed
        :param fruits: dict with the following fields
             - number: int, number of fruits in the environment
             - size: float, size of the fruits
             - reap: bool, whether another fruit object reappears when one is consumed
        :param agent: the agent evolving in the environment
        :param display: bool, whether to display the task or not
        """

        # Save the arguments for reset
        self.parameters = kwargs

        if not kwargs['map']:
            self.mapp_ = False

        self.done = False
        self.t = 0
        self.horizon = kwargs['horizon']
        self.width, self.height = kwargs['shape']
        self.display = kwargs['display']
        if self.display:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.screen.set_alpha(None)
        else:
            self.screen = pygame.Surface((self.width, self.height))
            self.screen.set_alpha(None)
        self.clock = pygame.time.Clock()
        
        self.npimage = np.zeros((self.width, self.height, 3))

        # Set a surface to compute Sensors

        # Initialize pymunk space
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.space.collision_slop = 0
        self.space.collision_persistence = 1
        self.space.collision_bias = 0
        self.handle_collisions()

        # Define the external walls
        texture_params = kwargs['walls_texture']
        self.obstacles = [
            Obstacle(
                shape='rectangle',
                position=(self.width/2, 5),
                angle=0,
                texture=texture_params,
                environment=self,
                length=self.width,
                width=20
            ),
            Obstacle(
                shape='rectangle',
                position=(self.width / 2, self.height-5),
                angle=0,
                texture=texture_params,
                environment=self,
                length=self.width,
                width=20
            ),
            Obstacle(
                shape='rectangle',
                position=(5, self.height / 2),
                angle=math.pi/2,
                texture=texture_params,
                environment=self,
                length=self.height,
                width=20
            ),
            Obstacle(
                shape='rectangle',
                position=(self.width - 5, self.height / 2),
                angle=math.pi/2,
                texture=texture_params,
                environment=self,
                length=self.height,
                width=20
            )
        ]

        # Add obstacles
        if not kwargs['map']:
            for obstacle_params in kwargs['obstacles']:
                obstacle_params['environment'] = self
                obstacle = Obstacle(**obstacle_params)
                self.obstacles.append(obstacle)

        if kwargs['map']:
            #create map object and check for connectivity
            self.mapp_ = Dungeons(space_size=(self.width, self.height), n_rooms=kwargs['n_rooms'])
            while not self.mapp_.topology.geom_type == 'Polygon':
                self.mapp_ = Dungeons(space_size=(self.width, self.height), n_rooms=kwargs['n_rooms'])
                print(self.mapp_.topology.geom_type)
            #arrange walls
            for wall in self.mapp_.walls:
                self.obstacles.append(
                Obstacle(
                    shape='rectangle',
                    position=(wall.x, self.height - wall.y),
                    angle=0,
                    texture=wall.texture,
                    environment=self,
                    width=wall.height,
                    length=wall.width
                    ))

        # Define the episode mode
        self.mode = kwargs['mode']

        # Create the goal in goal mode
        if self.mode == 'goal':
            self.goal_size = kwargs['goal']['size']
            self.goal = self.create_goal(kwargs['goal']['position'])
            if 'goal' not in kwargs['agent']['measurements']:
                kwargs['agent']['measurements'].append('goal')

        # Make sure we have the right measurements in survival mode
        if self.mode == 'survival':
            if 'health' not in kwargs['agent']['measurements']:
                kwargs['agent']['measurements'].append('health')
            if 'dead' not in kwargs['agent']['measurements']:
                kwargs['agent']['measurements'].append('dead')

        # Create the poisons
        self.poison_params = kwargs['poisons'].copy()
        self.poisons = []
        self.poison_params['environment'] = self
        self.poison_params['collision_type'] = 3
        if self.poison_params['positions'] == 'random':
            positions = ['random'] * self.poison_params['number']
            if kwargs['map']:
                positions = self.mapp_.generate_random_point(self.poison_params['number'])
        else:
            positions = self.poison_params['positions']
        for position in positions:
            poison = Edible(position=position, **self.poison_params)
            self.poisons.append(poison)

        # Once the poisons have been created, we switch to random position for the new poisons
        self.poison_params['position'] = 'random'

        # Create the fruits
        self.fruit_params = kwargs['fruits'].copy()
        self.fruits = []
        self.fruit_params['environment'] = self
        self.fruit_params['collision_type'] = 2
        if self.fruit_params['positions'] == 'random':
            positions = ['random'] * self.fruit_params['number']
            if kwargs['map']:
                positions = self.mapp_.generate_random_point(self.fruit_params['number'])
        else:
            positions = self.fruit_params['positions']
        for position in positions:
            fruit = Edible(position=position, **self.fruit_params)
            self.fruits.append(fruit)

        # Once the fruits have been created, we switch to random position for the new fruits
        self.fruit_params['position'] = 'random'

        # Add the agent
        self.agent_param = kwargs['agent'].copy()
        if kwargs['map']:
            self.agent_param['position'] = self.mapp_.generate_random_point()[-1]
        #print(self.agent_param['position'])
        self.agent = Agent(environment=self, **self.agent_param )
        
        # Set a surface to compute Sensors
        # TODO: change when multiple body parts
        self.sizeAroundAgent = max([ x.fovRange for x in self.agent.sensors]) + self.agent.radius
        
        self.agent.update_state()

    def create_goal(self, position):
        inertia = pymunk.moment_for_circle(1, 0, self.goal_size, (0, 0))
        goal = pymunk.Body(1, inertia)
        c_shape = pymunk.Circle(goal, self.goal_size)
        c_shape.elasticity = 1.0
        if position == 'random':
            position = (
                    random.randint(self.goal_size, self.width - self.goal_size),
                    random.randint(self.goal_size, self.height - self.goal_size),
            )
        goal.position = position
        c_shape.color = THECOLORS["green"]
        c_shape.collision_type = 4
        self.space.add(goal, c_shape)
        return goal

    def reload_screen(self):
        # Fill the screen
        self.screen.fill(THECOLORS["black"])
        
        # Do 10 mini-timesteps in pymunk for 1 timestep in our environment
        for _ in range(10):
            self.space.step(1. / 10)
        
        # Draw the entities
        self.draw_environment()
        
        # Get top view image of environment
        # TODO : Chhaaaaanggeeee  meeee! Something faaasteeeer!
        data = pygame.image.tostring(self.screen, 'RGB')
        pil_image = Image.frombytes('RGB', (self.width, self.height), data)
        image = np.asarray(pil_image.convert('RGB'))
        self.npimage = image
                        
        # Draw the agent
        self.agent.draw()

        data = pygame.image.tostring(self.screen, 'RGB')
        pil_image = Image.frombytes('RGB', (self.width, self.height), data)
        import os, os.path
        self.iter = len([name for name in os.listdir('images/')])
        image = np.asarray(pil_image.convert('RGB'))[20:84,20:84,:]
        image.setflags(write=1)
        """im_final = []
        for i in range(len(image)):
            if sum(image[i])!=0:"""
        pil_image = Image.fromarray(image)
        #print(image,'here')
        if np.all(image==0):
            print('yo')
            pass
        else:
            pil_image.save('images/'+str(self.iter+1)+'.png')
        # Update the display
        if self.display:
            pygame.display.flip()
        self.clock.tick()

    def handle_collisions(self):

        def begin_fruit_collision(arbiter, space, *args, **kwargs):

            # Remove the previous shape
            shapes = arbiter.shapes
            for shape in shapes:
                if shape.collision_type == 2:
                    self.fruits.remove(shape.body.entity)
                    space.remove((shape, shape.body))

                    # Update the measurements
                    self.agent.update_meas('items', 1)
                    self.agent.update_health(shape.body.entity.reward, self.mode)
                    self.agent.update_meas('fruits', 1)
                    self.agent.reward += shape.body.entity.reward

            if self.fruit_params['respawn']:
                if self.mapp_:
                    self.fruit_params['position'] = self.mapp_.generate_random_point()[-1]
                self.fruits.append(Edible(**self.fruit_params))

            return False

        def begin_poison_collision(arbiter, space, *args, **kwargs):

            # Remove the previous shape
            shapes = arbiter.shapes
            for shape in shapes:
                if shape.collision_type == 3:
                    self.poisons.remove(shape.body.entity)
                    space.remove((shape, shape.body))

                    # Update the measurements
                    self.agent.update_meas('items', 1)
                    self.agent.update_health(shape.body.entity.reward, self.mode)
                    self.agent.update_meas('poisons', 1)
                    self.agent.reward += shape.body.entity.reward

            if self.poison_params['respawn']:
                if self.mapp_:
                    self.poison_params['position'] = self.mapp_.generate_random_point()[-1]
                self.poisons.append(Edible(**self.poison_params))

            return False

        def begin_goal_collision(arbiter, space, *args, **kwargs):

            # This is the goal, we end the simulation and update the measurements
            self.agent.update_meas('goal', 1)
            self.agent.reward += 100

            return False

        def begin_wall_collision(arbiter, space, *args, **kwargs):

            # This is the goal, we end the simulation and update the measurements
            print('wall collision !')
            print(self.agent.body.position)

            #import pdb; 
            #pdb.set_trace()

            if self.agent.body.position[0]<=28:
                self.agent.body.position = (75,self.agent.body.position[1])
                print(self.agent.body.position)

            if self.agent.body.position[0]>=76:
                self.agent.body.position = (29,self.agent.body.position[1])

            if self.agent.body.position[1]<=28:
                self.agent.body.position = (self.agent.body.position[0],75)

            if self.agent.body.position[1]>=76:
                self.agent.body.position = (self.agent.body.position[0],29)


            return False

        fruit_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=2
        )
        fruit_collision_handler.begin = begin_fruit_collision

        poison_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=3
        )
        poison_collision_handler.begin = begin_poison_collision

        goal_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=4
        )
        goal_collision_handler.begin = begin_goal_collision

        wall_collision_handler = self.space.add_collision_handler(
            collision_type_a=0,
            collision_type_b=1
        )
        wall_collision_handler.begin = begin_wall_collision

    def step(self, action):
        """
        Method called to execute an action in the environment.
        :param action: string, the string code for the action to be executed by the agent
        :return: a tuple containing :
            - sensory_input : the sensory input at time t+1
            - reward: the reward at time t
            - done: whether the episode is over
            - measurements : the measurements at time t+1
        """

        self.t += 1

        # Default reward at time t
        self.agent.reward = -self.agent.living_penalty
        self.agent.update_health(-self.agent.living_penalty, self.mode)

        # Execute the action changes on the agent
        self.agent.apply_action(action)

        # Apply the step in the pymunk simulator
        self.reload_screen()

        # Get the agent's position and orientation
        x, y = self.agent.body.position
        print(self.agent.body.position)
        theta = self.agent.body.angle
        self.agent.set_meas('x', x)
        self.agent.set_meas('y', y)
        self.agent.set_meas('theta', theta)

        # Get the agent's perception
        for sensor in self.agent.sensors:
            sensor.get_sensory_input(self)

        # Look for termination conditions
        if self.mode == 'goal' and self.agent.meas['goal'] == 1:
            self.done = True
        if self.mode == 'survival' and self.agent.meas['dead'] == 1:
            self.done = True
        if self.t >= self.horizon - 1:
            self.done = True

        return self.agent.state, self.agent.get_reward(), self.done, self.agent.get_meas()

    def reset(self):
        self.parameters['agent'].update(self.agent.get_new_averages())
        for sensor in self.agent.sensors:
            sensor.reset()
        self.__init__(**self.parameters)

    def draw_environment(self):
        
        # TODO: Replace with entities

        # Draw the fruits
        for fruit in self.fruits:
            fruit.draw()

        # Draw the poisons
        for poison in self.poisons:
            poison.draw()

        for obstacle in self.obstacles:
            obstacle.draw()
        """
        # Draw the walls
        for wall in self.walls:
            wall.draw()

        # Draw the goal
        self.goal.draw()
        """
