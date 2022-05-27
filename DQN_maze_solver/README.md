## Deep Q-Network for Dynamic Maze Solving

### Files
- `COMP6247Maze20212022.npy`: Binary file containing maze to be trained on. Read by read_maze.py.
- `read_maze.py`: Reads binary maze and creates a python maze.
- `DQN_final.py` Reads functions containing maze information from `read_maze.py`. Performs training, validation, and testing of algorithm. Outputs the final path, and surrounding information and action at each step.
- `Maze_solver.h5`: Serialised model, can be read in to continue training, or used for inference.
- `output.json` JSON of each training step. Contains the current location, state of surrounding squares, and the action taken.
- `requirements.txt`: Packages required to compile code, automatically installed by `Makefile`.
- `Makefile` Used to create virtual environment, install dependancies and compile `DQN_final.py`.

### Compiling code
This folder contains a single python script to train, validate, and produce output from, a DQN dynamic maze solving algorithm.

To compile the code, simply run the `make` command from the terminal in the current directory. A virtual environment called venv will be created, and the required dependancies installed.

### Output
Whilst the `output.json` file given here may be considered an example of an output in the correct format, the solution provided is not a feasible one.
