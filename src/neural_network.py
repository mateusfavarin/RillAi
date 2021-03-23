from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow.config.threading as tf

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import random
import socket

# Getting a better performance from tensorflow
NUM_CORES = 6
tf.set_intra_op_parallelism_threads(NUM_CORES)
tf.set_inter_op_parallelism_threads(2)

# Reward Constants
END_OF_FRAME = 0xBA11
END_OF_GAME = 0xDEAD
DEAD_PLAYER = END_OF_GAME / 1201.0

# Hyper parameters of the neural network
REPLAY_MEMORY_SIZE = 4_000_000
MIN_REPLAY_MEMORY_SIZE = 200_000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 10_000
SAVE_MODEL_EVERY = 1
LEARNING_RATE = 0.00025
epsilon = 1
EPSILON_DECAY = 0.0000009
MIN_EPSILON = 0.01

# Game parameters
NUM_AIS = 4
NUM_FRAMES = 3
PLAYER_INFO = NUM_FRAMES + 1
BALL_INFO = 2
MAX_BALLS = 5
INPUT_SIZE = (BALL_INFO * MAX_BALLS * NUM_FRAMES) + PLAYER_INFO
OUTPUT_SIZE = 4

# Main loop variables
frames_update = 1
frames_played = 1
games_played = 0

# Plot parameters
NUM_GAMES_PLOT = 50
update_plot = False

# Setting matplotlib style
plt.style.use('dark_background')

def mean_list(v):
    total = 0
    for n in v:
        total += n
    return total / len(v)

def plot_data():
    with open("reward_data.txt") as file:
        data = file.readlines()
        y = [[], [], [], []]
        y_plot = [[], [], [], []]
        for d in data:
            d = d.split()
            for i in range(4):
                y[i].append(float(d[i]))

        i = 0
        while i + NUM_GAMES_PLOT <= len(y[0]):
            for j in range(4):
                y_plot[j].append(mean_list(y[j][i : i + NUM_GAMES_PLOT]))
            i += NUM_GAMES_PLOT

        x_plot = [i * NUM_GAMES_PLOT for i in range(1, len(y_plot[0]) + 1)]
        plt.plot(x_plot, y_plot[0], 'r', label="Rilla Roo")
        plt.plot(x_plot, y_plot[1], 'y', label="N. Brio")
        plt.plot(x_plot, y_plot[2], 'm', label="Coco")
        plt.plot(x_plot, y_plot[3], 'c', label="Koala Kong")
        plt.legend()
        plt.title("Average Moving Reward (50 Games)")
        plt.xlabel("Number of Games")
        plt.ylabel("Average Reward")
        plt.savefig("reward_per_game")
        plt.clf()

class Console:
    def __init__(self, host = '127.0.0.1', port = 8001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.connection, self.address = self.sock.accept()
        print('Python got a client at {}'.format(self.address))

    def recv(self):
        self.buffer = self.connection.recv(1024).decode()
        return self.buffer

    def send(self, msg):
        _ = self.connection.send(msg.encode())

    def close(self):
        _ = self.connection.close()


class DeepQNetwork:
    def __init__(self, input_size, output_size, learning_rate, filename):
        # Create main model
        self.model = self.create_model(input_size, output_size, learning_rate)
        # Create predict model using the same weights as the main model
        self.target_model = self.create_model(input_size, output_size, learning_rate)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.filename = filename

    def create_model(self, input_size, output_size, learning_rate):
        model = Sequential([
            InputLayer((input_size, )),
            Dense(2 * input_size, activation="relu"),
            Dense(2 * input_size, activation="relu"),
            Dense(output_size, activation="linear")
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return np.argmax(self.model(state), axis=1).item()

    def train(self, done, update_target):
        if (len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE):
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = []
        new_current_states = []
        for (current_state, _, _, new_current_state) in minibatch:
            current_states.append(current_state[0])
            new_current_states.append(new_current_state[0])

        current_states = np.array(current_states)
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array(new_current_states)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
            max_future_q = np.max(future_qs_list[index][0])
            new_q = reward + DISCOUNT * max_future_q

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state[0])
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), shuffle=False, verbose=0)

        if update_target:
            print("Target updated")
            self.target_model.set_weights(self.model.get_weights())

        if done:
            self.model.save(self.filename + ".model")

def convert_state(state):
    # Shuffle the balls so that every neuron gets trained in an equal distribution
    balls_order = [i for i in range(MAX_BALLS)]
    random.shuffle(balls_order)

    normalized_state = [[], [], [], []]

    i = 0
    j = 0
    frame = 0

    # while you didn't read the data of every frame
    while frame < NUM_FRAMES:
        while state[j] != END_OF_FRAME:
            j += 1

        # Normalize the position of the player
        normalized_state[0].append(state[i] / 1201.0)
        normalized_state[1].append(-1 * state[i + 1] / 1201.0)
        normalized_state[2].append(state[i + 2] / 1201.0)
        normalized_state[3].append(-1 * state[i + 3] / 1201.0)
        i += 4

        # Correct j for the last frame, since it contains cooldown data as well
        target = j if frame != NUM_FRAMES - 1 else j - NUM_AIS

        # Normalize the balls
        # Adding default balls at (0, 0)
        balls = []
        for _ in range(MAX_BALLS):
            balls.append([0.0, 0.0])

        # Adding the ball information in reverse to preserve its order in the neurons
        k = 0
        y = target - 1
        while i < y:
            balls[k] = [state[y - 1] / 2500.0, state[y] / 2500.0]
            k += 1
            y -= 2

        i = target

        # Group balls in shuffled order while preserving the neuron location
        for k in range(MAX_BALLS):
            normalized_state[0] += balls[balls_order[k]]
            balls[balls_order[k]][0] *= -1
            balls[balls_order[k]][1] *= -1
            normalized_state[1] += balls[balls_order[k]]
            temp = balls[balls_order[k]][0]
            balls[balls_order[k]][0] = balls[balls_order[k]][1] * -1
            balls[balls_order[k]][1] = temp
            normalized_state[2] += balls[balls_order[k]]
            balls[balls_order[k]][0] *= -1
            balls[balls_order[k]][1] *= -1
            normalized_state[3] += balls[balls_order[k]]

        # If you're gathering data of the last frame
        if frame == NUM_FRAMES - 1:
            # Get cooldown data
            while i < j:
                normalized_state[0].append(state[i])
                normalized_state[1].append(state[i + 1])
                normalized_state[2].append(state[i + 2])
                normalized_state[3].append(state[i + 3])
                i += 4

        i += 1
        j += 1
        frame += 1

    for i in range(NUM_AIS):
        normalized_state[i] = np.array([normalized_state[i]])

    return normalized_state

agent = DeepQNetwork(INPUT_SIZE, OUTPUT_SIZE, LEARNING_RATE, "rillai-kick")
#agent.model = load_model("crashball.model")
#agent.target_model.set_weights(agent.model.get_weights())
console = Console()
first_connection = True
update_target = False
action = [0, 0, 0, 0]

while frames_played < 10_000_000:

    if first_connection:
        status = console.recv()
        if (status != "Connected"):
            print("ERROR: Couldn't connect to the lua script.")
            exit()
        first_connection = False

    current_state = list(map(float, console.recv().split()))
    current_state = convert_state(current_state)

    done = False
    while not done:
        if update_plot:
            plot_data()
            update_plot = False

        for i in range(NUM_AIS):
            if np.random.random() > epsilon:
                action[i] = agent.get_qs(current_state[i])
            else:
                action[i] = np.random.randint(0, OUTPUT_SIZE)

        action_str = ""
        for act in action:
            action_str = action_str + str(act) + " "

        console.send(str(action_str)+"\n")
        reward = list(map(float, console.recv().split()))
        console.send("Reward received\n")
        done = True if reward[0] == END_OF_GAME else False

        agent.train(done, update_target)

        if done:
            games_played += 1
            if games_played % NUM_GAMES_PLOT == 0:
                update_plot = True
            print("Frames played: "+str(frames_played))
            print("Epsilon: "+str(epsilon))
            break

        new_state = list(map(float, console.recv().split()))
        new_state = convert_state(new_state)

        for i in range(NUM_AIS):
            if current_state[i][0][0] != DEAD_PLAYER:
                agent.update_replay_memory((current_state[i], action[i], reward[i], new_state[i]))

        current_state = new_state
        frames_played += 1
        frames_update += 1
        update_target = False

        if frames_update > UPDATE_TARGET_EVERY:
            frames_update = 0
            update_target = True

        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)