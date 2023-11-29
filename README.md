# DnD Combat encounters AI

Developing an RL agent to play DnD combat.

## Project structure
* `agents/` : a couple of trained agents for testing purposes
* `notebooks/` : various Jupyter notebooks used during RnD process, including tests of game mechanics, notebooks for agent training and comparison
* `src/` : Source code in python
    - `src/agent/` : Code regarding RL agents, neural networks and functions for training / evaluation
    - `src/dnd/` : Game simulation and related utility functions
    - `src/gui/` : Code used for GUI implementation
    - `src/utils/` : commonly used utility and plotting functions 
* `Tokens/` : Images and .json files describing DnD game players/units
* `DnD.ipynb` : notebook containing the GUI implementation
* `DnD.py` : GUI implementation in .py script. It is the main script for app on the [huggingface](https://huggingface.co/spaces/DnD-inc/DnD)
