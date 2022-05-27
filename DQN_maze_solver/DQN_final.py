import numpy as np
import json
from read_maze import get_local_maze_information
from read_maze import load_maze
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU

load_maze()

# Define Classes ----------------------------


class State(object):
    """
        Given an x and y coordinate, define a class holding attributes about the corresponding state.
        Input: x, y: Numerical coordinates 1 <=/=> 199
        Returns: State object
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.URDL_idx = [(0, 1), (1, 2), (2, 1), (1, 0)]
        self.information = get_local_maze_information(x, y)

    def aroundNoCorners(self):
        """
            Remove corners of state object, flatten into an array of the form:
                [Up, Up(fire), Right, Right(fire), Down, Down(fire), Left, Left(fire), x, y]
            Input: None
            Returns: NdArray of size (1, 10)
        """

        # Flatten input
        around_flat = np.array(self.information.flatten(),
                               dtype=int).reshape(1, -1)
        # Remove corner elements. Reorder adjacent elements: URDL, append x, y to the end
        around_flat_no_corners = np.append(
            around_flat[0][[2, 3, 10, 11, 14, 15, 6, 7]], [self.x, self.y])

        return np.array([list(around_flat_no_corners)])

    def validA(self):
        """
            Input: None
            Returns: NdArray of actions not obstructed by walls
        """

        return np.flatnonzero(np.array([self.information[idx][0] for idx in self.URDL_idx]) != 0)

    def walls(self):
        """
            Input: None
            Returns: NdArray of actions obstructed by walls
        """

        return np.flatnonzero(np.array([self.information[idx][0] for idx in self.URDL_idx]) == 0)

    def fires(self):
        """
            Input: None
            Returns: Dictionary of length 4, key i maps to fire time remaining for i in actions
        """

        return {i: self.information[j][1] for i, j in enumerate(self.URDL_idx)}

    def location(self):
        """
            Input: None
            Returns: NdArray of State coordinates in correct format for DQN
        """

        return np.array(list(np.array([self.x, self.y]).reshape(1, -1)))


class Experience(object):
    """
        Class of experience buffer used to store previous episodes
        Inputs:
            model: DQN model object
            max_memory: Max length of memory. once reached, old episodes will be dropped when new ones are added
            data_size: Number of previous episodes to use per training step
            discount: Gamma in Q(s, a). Used to weight present vs future reward
    """

    def __init__(self, model, max_memory, data_size=10, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.data_size = data_size
        self.discount = discount
        self.memory = []

    def remember(self, episode):
        """
            Adds episode to experience buffer memory
            Input: episode: list containing information about a previous episode. Of the form:
                [starting location,
                 action taken,
                 reward,
                 end_location,
                 boolean indicating whether the game ended]
            Returns: None
        """

        self.memory.append(episode)

        # If the memory is full, drop the first episode
        if len(self.memory) > self.max_memory:
            del self.memory[0]

        return

    def predict(self, envstate):
        """
            Predict expected reward of all actions given a state
            Input: envstate: State object
            Returns: (1, 4) array of expected returns for actions 0 to 3
        """

        return self.model.predict(envstate)[0]

    def get_data(self):
        """
            Extract training inputs and targets from experience replay buffer
            Inputs: None
            Returns: 
                (data_size, 2) array of states to input to the DQN
                (data_size, 4) array of targets corresponding to actions 0 to 3
        """

        env_size = 2
        mem_size = len(self.memory)
        sample_size = min(mem_size, self.data_size)
        inputs = np.zeros((sample_size, env_size))
        targets = np.zeros((sample_size, 4))

        # Choose data_size - 1 episodes from memory for training. Append most recent epsiode to ensure it's included
        for i, j in enumerate(np.append(np.random.choice(range(mem_size), sample_size - 1, replace=False), -1)):
            location, action, reward, location_next, game_over = self.memory[j]
            inputs[i] = location

            # Q_sa = max_a' Q(s', a')
            Q_sa = np.max(self.predict(location_next))
            
            if game_over:
                targets[i, action] = reward
            else:
                # Q(s, a) = reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets

# Define helper functions ----------------------------


def move(x, y, a):
    """
        Input:
            x, y: Numerical coordinates 1 <=/=> 199
            a: action index 0 <=/=> 3
        Returns: Tuple of new x and y coordinates after taking action a
    """

    if a == 0:
        x -= 1
    elif a == 1:
        y += 1
    elif a == 2:
        x += 1
    elif a == 3:
        y -= 1

    return x, y


def actionWord(a):
    """
        Input: a: action index 0 <=/=> 3
        Returns: Human-readable string corresponding to a given action
    """
    if a == 0:
        word = "up"
    elif a == 1:
        word = "right"
    elif a == 2:
        word = "down"
    elif a == 3:
        word = "left"

    return word

# -----------------------------------------


def qtrain(model, epsilon, max_memory=500):
    """
        Perform a single training iteration of the DQN model
        Input:
            model: DQN model object
            epsilon: Parameter defining probability of taking a random action (see epsilon greedy algorithm)
        Returns: Dictionary of results where entry i is the location, surroundings, and action taken at step i
    """

    results = {}

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    game_over = False

    x, y = (1, 1)
    x_prev, y_prev = (1, 1)

    episode_num = 0

    while not game_over:

        envstate = State(x, y)

        print(envstate.location())

        # If fire is detected, wait in the same place until it's gone
        while sum(envstate.fires().values()) > 0:
            envstate = State(x, y)
            
            results[str(episode_num)] = {"location": envstate.location(
            ), "around": envstate.information, "action": "wait"}
            
            episode_num += 1

        # Loop through surrounding walls, adding the corresponding episode to memory
        # This is time effective as we don't need to check the next state; the agent doesn't move!
        for wall_a in envstate.walls():
            experience.remember(
                [envstate.location(), wall_a, -10, envstate.location(), False])

        valid_actions = envstate.validA()

        #  If there's only one valid action, take it
        if len(valid_actions) == 1:
            action = valid_actions[0]

        else:
            # If a random uniform number is below epsilon, choose a random action
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_actions)
                
                # If the randomly chosen action leads the agent backwards, choose another one
                while move(x, y, action) == (x_prev, y_prev):
                    action = np.random.choice(valid_actions)

            else:
                # Get the expected Q-values and preferred action from the DQN
                NN_out = experience.predict(envstate.location())
                action = np.argmax(NN_out)
                
                # If the chosen action is invalid or backtracking, choose the next best one
                while action not in valid_actions or move(x, y, action) == (x_prev, y_prev):
                    NN_out[np.argmax(NN_out)] = -np.inf
                    action = np.argmax(NN_out)

        # Move to the next state
        x_prev, y_prev = x, y
        x, y = move(x, y, action)

        if x == 199 and y == 199:
            reward = 100
            game_over = True
        else:
            reward = -0.1

        next_envstate = State(x, y)

        # If the agent arrives at a deadend, large penalty
        if len(next_envstate.walls()) >= 3 and not game_over:
            reward = -10

        results[str(episode_num)] = {"location": envstate.location(
        ), "around": envstate.information, "action": actionWord(action)}
        episode_num += 1

        # Store previous episode in memory
        episode = [envstate.location(), action, reward,
                   next_envstate.location(), game_over]
        experience.remember(episode)

        if episode_num % 100 == 0:
            print("Episodes: ", episode_num, "Location: ", envstate.location())

        # Get input and output data for training
        inputs, targets = experience.get_data()

        # Fit DQN model
        model.fit(
            inputs,
            targets,
            epochs=10,
            batch_size=1,
            verbose=0
        )

    return results


def build_model(lr=0.01):
    """
        Build a keras DQN model object
        Input: lr: Learning rate for training DQN
        Returns: keras DQN model object
    """

    model = Sequential()

    model.add(layers.Input(shape=(2,)))
    model.add(layers.Dense(199**2 / 2))
    model.add(PReLU())
    model.add(layers.Dense(4))

    opt = keras.optimizers.Adam(learning_rate=lr, clipnorm=1)
    model.compile(optimizer=opt, loss='mse')

    return model

def train_DQN(model):
    """
        Train the DQN model until path convergence
        Inputs:
            model: DQN model object
            lr: Learning rate
        Returns: Dictionary of the shortest path found during training.
                 Contains location, surroundings, and actions taken at each step
    """

    # Define test matrix for pseudo-validation
    test_matrix = np.zeros((199, 199, 2))
    for i in range(1, 200):
        for j in range(1, 200):
            test_matrix[i - 1][j - 1] = (i, j)
    test_matrix = test_matrix.reshape((199**2, 2))

    total_episodes = 0

    prev_actions = [0] * 199**2
    actions = [-1] * 199**2

    best_results = {i: None for i in range(1000)}

    game_num = 0

    while not np.array_equal(prev_actions, actions):

        print("Game Number:", game_num)

        epsilon = 1

        epoch_results = qtrain(model, epsilon)

        total_episodes += len(epoch_results)

        predictions = model.predict(test_matrix)
        actions = [np.argmax(prediction) for prediction in predictions]

        # If the recently found path is shorter than the current best, update
        if len(epoch_results < len(best_results)):
            best_results = epoch_results

        prev_actions = actions

        # Save the current model every 10 games
        if game_num % 10 == 0:
            model.save("Maze_solver.h5")

        # Decay epsilon by 0.05 each game
        epsilon -= 0.05 if epsilon >= 0.05 else 0

        game_num += 1
        
    model.save("Maze_solver.h5")
    
    return best_results


model = build_model()

best_results = train_DQN(model)

# Read saved model back into object
model = keras.models.load_model("Maze_solver.h5", compile=True)

x_test, y_test = (1, 1)
test_episode_num = 0

results_test = {}

while not (x_test == 199 and y_test == 199) and len(results_test) <= len(best_results):

    state = State(x_test, y_test)
    # Get the expected Q-values from the DQN
    expected_rewards = model.predict(state.location())[0]

    actions = np.argsort(expected_rewards)
    action_idx = -1

    # If the action chosen by the DQN is invalid, choose the next best one
    while actions[action_idx] not in state.validA():
        action_idx -= 1

    action = actions[action_idx]

    fire_time = state.fires()[action]
    
    # If the chosen action has no fire, take that action. If there is fire, add the expected penalty for waiting
    # for the fire. If the action is still best, wait, else: change direction to the next best action
    if fire_time == 0:
        results_test[str(test_episode_num)] = {"location": state.location()[0].tolist(
        ), "around": state.information.tolist(), "action": actionWord(action)}
        x_test, y_test = move(x_test, y_test, actions[action_idx])
    else:
        expected_reward_after_wait = expected_rewards[action] + \
            fire_time * -0.1
        potential_action = actions[action_idx - 1]
        if expected_reward_after_wait <= expected_rewards[potential_action] and potential_action in state.validA():
            results_test[str(test_episode_num)] = {"location": state.location()[0].tolist(
            ), "around": state.information.tolist(), "action": actionWord(potential_action)}
            x_test, y_test = move(x_test, y_test, potential_action)
        else:
            while state.fires()[action] != 0:
                results_test[str(test_episode_num)] = {"location": state.location()[
                    0].tolist(), "around": state.information.tolist(), "action": "wait"}
                state = State(x_test, y_test)

    test_episode_num += 1

# Output the results dictionary to a JSON file.
with open("output.json", "w") as outfile:
    if len(results_test) <= len(best_results):
        json.dump(results_test, outfile, indent=4)
    else:
        json.dump(best_results, outfile, indent=4)
