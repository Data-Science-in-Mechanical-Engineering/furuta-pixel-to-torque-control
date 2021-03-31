# Pixel-To-Torque Policy Learning in hardware with the Quanser QUBE-Servo 2

This repository contains scripts to reproduce a pixel-to-torque policy learning pipeline on a vision-based Furuta pendulum. 

The repository is structured as follows:

**Vision to State**: Collecting data on the hardware system and training a state estimator 
**State to Input**: Training a PPO reinforcement learning agent from encoder states or from a state estimator

To train the agents, we use [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and Pytorch.

You can reproduce the learning pipeline on your own vision-based Furuta pendulum. Detailed instructions on how to reproduce our setup can be found in the [Quanser Driver](TODO) repository.

## Installation and Setup
Python version >=3.6 is needed. Clone this repository and install requirements with 

```
pip3 install -r requirements.txt 
``` 

You also need to install the [Quanser Driver](TODO) repository. Please follow instructions there.

Finally add package/ directory to PYTHONPATH with

```
export PYTHONPATH="${PYTHONPATH}:/path/to/package"
```

## Usage

Example demonstrations can be found in package/main_*.py. Code is commented.

### License

Copyright © 2017, Max Planck Insitute for Intelligent Systems.

Authors: Steffen Bleher

Released under the [MIT License](LICENSE).