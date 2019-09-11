# Flatland

**Flatland** is a 2D learning environment. (photo) 

It is a suite of 2D navigation and puzzle-solving tasks for learning agents. 

Its purpose is to act as a testbed for
research in artificial intelligence (especially reinforcement learning) and robotics. It is inspired from [DeepMind Lab](https://github.com/deepmind/lab) which is a suite of 3D navigation and puzzle-solving tasks for learning agents, and [Matt Harvey's series on Reinforcement Learning](https://github.com/harvitronix/reinforcement-learning-car).

## About

Disclaimer: This is not an official Softbank Robotics Europe product.

If you use *Flatland* in your research and would like to cite
the *Flatland* environment, we suggest you cite the [Flatland paper](?).

Feel free to open issues if you have questions.

## Installation on Linux

### Clone Flatland, e.g. by running

```shell
$ git clone https://github.com/?
$ cd ?
```

### Install requirements

- Install Pygame

Install Pygame's dependencies with:

`sudo apt install mercurial libfreetype6-dev libsdl-dev libsdl-image1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libportmidi-dev libavformat-dev libsdl-mixer1.2-dev libswscale-dev libjpeg-dev`

Then install Pygame itself:

`pip3 install hg+http://bitbucket.org/pygame/pygame`

- Install Pymunk

`pip install pymunk`

## Pre-defined environments and tasks

Using Flatland you can create different type of environments. Some of them are predefined:

- *Survival* is a 2D game where the agent tries to eat the most fruits (orange dots) while avoiding poisons (purple dots) in order to survive.
- *Navigation-I* is a 2D navigation problem in a maze where the agent needs to find a fixed goal.
- *Navigation-II* is a 2D navigation problem in a maze where the agent needs to find a random goal.

[Here](?) you can find a complete description of these pre-defined games. (photo)

## Getting started in 2 lines of Python

Flatland works with the same type of API defined in [OpenAI Gym](https://gym.openai.com/). Hence, once the environment instantiated, you can use the environment with the same methods.

Example: 

```python
    import flatland
    env = flatland.make('Survival')
    env.reset()
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample()) # take a random action
```

[GIF of result]

## Specifying you own environment and task

Environments and tasks can be defined using json configurations such as this one:

```python
game_parameters = {
        "display": True,
        "horizon": 10000,
        "shape": (900, 600),
        "mode": "survival",
        "goal": {
            "position": "random",
            "size": 30.
        },
        "walls": {
            "number": 2,
            "position": [(300, 300), (650, 300)],
            "size": 100
        },
        "poisons": {
            "number": 10,
            "position": "random",
            "size": 10
        },
        "fruits": {
            "number": 25,
            "position": "random",
            "size": 10
        },
        "agent": {
        "living_penalty":1,
        "pos": (100,100),
        "angle": 0,
        "sensor": {
            "type": "image",
            "resolution": 64,
            "range": 300,
            "angle": math.pi * 110 / 180,
            "spread": 5,
            "display": False
        },
        "actions": ["forward", "turn_left", "turn_right"],
        "measurements": ["health", "fruits", "poisons"]
    }
}
```

Please refer to the [documentation](?) for all possible options in the json configuration.
    
    

