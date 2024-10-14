"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm

    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""
import pickle
import sys
# Change to the path of your ZMQ python API
import numpy as np
import tensorflow as tf
from numpy import random



import numpy as np
import tensorflow as tf
from collections import deque

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class DQNAgent:
    def __init__(self, state_size, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.7, update_target_frequency=5):
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_frequency = update_target_frequency
        self.update_target_counter = 0
        self.learning_rate = learning_rate

        # Q-networks
        self.model = QNetwork(state_size, num_actions)
        self.target_model = QNetwork(state_size, num_actions)
        self.target_model.set_weights(self.model.get_weights())

        # Experience replay
        self.memory = deque(maxlen=2000)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.model.predict(np.expand_dims(state, axis=0))[0]
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state):
        # Store experience in replay memory
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size, fileObj):
        if len(self.memory) < batch_size:
            return

        # Sample a batch from replay memory
        samples = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in samples:
            state, action, reward, next_state = self.memory[idx]
            # Calculate the target Q-values
            target = self.target_model.predict(np.expand_dims(state, axis=0))[0]
            Q_future = np.max(self.target_model.predict(np.expand_dims(next_state, axis=0))[0])
            max_future_q = reward[0] + self.gamma * Q_future
            current_q_value = abs(target[0,action])
            target[0,action] = (1 - self.learning_rate) * target[0,action] + self.learning_rate * max_future_q
            # Update the Q-network
            td_error = np.abs(max_future_q - current_q_value)
            print(f'TD Error: {td_error}')
            fileObj.write(f'TD Error: {td_error}\n')
            self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(target, axis=0), epochs=1, verbose=0)

        # Update target network periodically
        self.update_target_counter += 1
        if self.update_target_counter % self.update_target_frequency == 0:
            print("Copying the weights to target network")
            self.update_target_network()

    def update_target_network(self):
        # Copy weights from the Q-network to the target network
        self.target_model.set_weights(self.model.get_weights())


maxBoxes = float('-inf')
sys.path.append('/Applications/coppeliaSim.app/Contents/Resources/programming/zmqRemoteApi/clients/python/zmqRemoteApi')

alpha = 0.6
gamma = 0.4
epsilon = 0.3

from zmqRemoteApi import RemoteAPIClient


class Simulation():
    def __init__(self, sim_port=23004):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')

        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)

        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()

    def getObjectHandles(self):
        self.tableHandle = self.sim.getObject('/Table')
        self.boxHandle = self.sim.getObject('/Table/Box')

    def dropObjects(self):
        self.blocks = 18
        frictionCube = 0.06
        frictionCup = 0.8
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript, self.tableHandle)
        self.client.step()
        retInts, retFloats, retStrings = self.sim.callScriptFunction('setNumberOfBlocks', self.scriptHandle,
                                                                     [self.blocks],
                                                                     [massOfBlock, blockLength, frictionCube,
                                                                      frictionCup], ['cylinder'])

        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue = self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    def getObjectsInBoxHandles(self):
        self.object_shapes_handles = []
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step

    def action(self, direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir * span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()


def getMappings(positions):
    d = {}
    for index in range(len(positions)):
        col = -1
        if index < 9:
            col = 0
        else:
            col = 1
        x = positions[index][0]
        y = positions[index][1]
        if ((x > -0.05 and x < (-0.05 + (0.1) / 3)) and (y < 0.05 and y > (0.05 - (0.1 / 2)))):
            if 1 not in d:
                d[1] = 1
            else:
                d[1] += 1
            continue
        if ((x > (-0.05 + (0.1) / 3) and x < (-0.05 + 2 * (0.1) / 3)) and (y < 0.05 and y > (0.05 - (0.1 / 2)))):
            if 2 not in d:
                d[2] = 1
            else:
                d[2] += 1
            continue
        if ((x > (-0.05 + 2 * (0.1) / 3) and x < (-0.05 + 3 * (0.1) / 3)) and (y < 0.05 and y > (0.05 - (0.1 / 2)))):
            if 3 not in d:
                d[3] = 1
            else:
                d[3] += 1
            continue
        if ((x > -0.05 and x < (-0.05 + (0.1) / 3)) and (y < (0.05 - (0.1 / 3)) and y > (0.05 - (2 * 0.1 / 3)))):
            if 4 not in d:
                d[4] = 1
            else:
                d[4] += 1
            continue
        if ((x > (-0.05 + (0.1) / 3) and x < (-0.05 + 2 * (0.1) / 3)) and (
                y < (0.05 - (0.1 / 3)) and y > (0.05 - (2 * 0.1 / 3)))):
            if 5 not in d:
                d[5] = 1
            else:
                d[5] += 1
            continue
        if ((x > (-0.05 + 2 * (0.1) / 3) and x < (-0.05 + 3 * (0.1) / 3)) and (
                y < (0.05 - (0.1 / 3)) and y > (0.05 - (2 * 0.1 / 3)))):
            if 6 not in d:
                d[6] = 1
            else:
                d[6] += 1
            continue
        if ((x > -0.05 and x < (-0.05 + (0.1) / 3)) and (y < (0.05 - (2 * 0.1 / 3)) and y > (0.05 - (3 * 0.1 / 3)))):
            if 4 not in d:
                d[4] = 1
            else:
                d[4] += 1
            continue
        if ((x > (-0.05 + (0.1) / 3) and x < (-0.05 + 2 * (0.1) / 3)) and (
                y > (0.05 - (3 * 0.1 / 3)))):
            if 5 not in d:
                d[5] = 1
            else:
                d[5] += 1
            continue
        if ((x > (-0.05 + 2 * (0.1) / 3) and x < (-0.05 + 3 * (0.1) / 3)) and (
                y > (0.05 - (3 * 0.1 / 3)))):
            if 9 not in d:
                d[9] = 1
            else:
                d[9] += 1
            continue
    print(d)
    s = 0
    for k in d:
        s += d[k]
        global maxBoxes
        maxBoxes = max(maxBoxes, d[k])
    if (s != 18):
        print("badddd")


def buildQTable(rows, columns):
    q = []
    for i in range(rows):
        t = []
        for j in range(columns):
            t.append(0)
        q.append(t)
    return q


states = []


def buildStates(l, sum):
    if (sum == 0 and len(l) == 4):
        global states
        states.append(l.copy())
    if (sum < 0 or len(l) == 9):
        return
    else:
        for i in range(0, 10):
            l.append(i)
            buildStates(l, sum - i)
            l.pop()


buildStates([], 9)


class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_values = np.zeros((num_states, num_actions))

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            print("Performing Random Action - Exploration:-")
            # Explore: choose a random action
            return random.randint(0, self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            ind = np.argmax(self.q_values[state, :])
            if(self.q_values[state, ind]==0):
                print(f"Exploitation Random:- {ind}")
                return random.randint(0, self.num_actions)

            else:
                print(f"Exploitation Q-Table:- {ind}")
                return ind

    def update_q_values(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_values[next_state, :])
        target = reward + self.discount_factor * self.q_values[next_state, best_next_action]
        p = self.q_values[state, action]
        self.q_values[state, action] = (1 - self.learning_rate) * self.q_values[
            state, action] + self.learning_rate * target
        print(f"Updating the value of state:action in qtable-->{state}:{action} from {p} to-> {self.q_values[state, action]}")


def mapStates(li):
    d = {}
    l = []
    for k in range(len(li)):
        for j in range(len(li)):
            l.append(''.join(map(str, li[k])) + ''.join(map(str, li[j])))
    for i in range(len(l)):
        d[l[i]] = i
    return d


def getState(positions):
    l = []
    for i in range(0, 4):
        t = []
        for j in range(2):
            t.append(0)
        l.append(t)
    for index in range(len(positions)):
        if index < 9:
            col = 0
        else:
            col = 1
        x = positions[index][0]
        y = positions[index][1]
        if (0.05 > y > 0) and (-0.05 < x < 0):
            l[0][col] += 1
        if (0.05 > y > 0) and (0 < x < 0.05):
            l[1][col] += 1
        if (0 > y > -0.05) and (-0.05 < x < 0):
            l[2][col] += 1
        if (0 > y > -0.05) and (0 < x < 0.05):
            l[3][col] += 1
    for i in range(2):
        cn = 0
        for k in range(4):
            cn += l[k][i]
        if (cn != 9):
            print("Invalid State!! -> Object fell out of container!!! Resetting the Simulation")
            return -1

    st = ""
    for i in range(2):
        for k in range(4):
            st += str(l[k][i])
    return st


import math

def write_q_table_to_txt(q_table, file_path):
    with open(file_path, 'w') as file:
        # Write the Q-table header
        file.write("State/Action\t\tQ-Value\n")

        # Iterate through each row (state) in the Q-table
        for i in range(q_table.shape[0]):
            for j in range(q_table.shape[1]):
                state_action = f"({i}, {j})"
                q_value = q_table[i, j]
                file.write(f"{state_action}\t\t{q_value}\n")
def calculateReward(state):
    counts = list(map(int, list(state)))
    s = 0
    for i in range(4):
        s += abs(counts[i] - counts[i + 4])
    d = 0
    for i in range(8):
        d += abs(counts[i]-2)
    return [max(0, 60 - s - d), s, d]

import time
def main():
    episodes = 30
    logFile = 'log.txt'
    modelPath = 'finalModel.h5'
    logFileObj = open(logFile, 'w')
    steps_to_update_target_model = 0
    highestReward = float('-inf')
    steps = 90  # changed this to run for 20 steps.
    # qAgent = QLearningAgent(48400, 4, 0.3, 0.9, 0.1)
    agent = DQNAgent(48400, 4)
    total_reward = 0
    global states
    mappedStates = mapStates(states)
    for episode in range(episodes):
        print(f'Running episode: {episode + 1}')
        env = Simulation()
        positions = env.getObjectsPositions()
        state = getState(positions)
        total_reward = 0
        while state==-1:
            env.stopSim()
            time.sleep(2)
            env = Simulation()
            positions = env.getObjectsPositions()
            state = getState(positions)
        highestReward = float('-inf')
        state = np.reshape(list(map(int, state)), [1, 8])
        for _ in range(steps):
            steps_to_update_target_model+=1
            # Select action
            action = agent.select_action(state)

            # Execute action in the environment
            env.action(direction=env.directions[action])
            positions = env.getObjectsPositions()
            next_state = getState(positions)
            if next_state == -1:
                print("invalid state!!")
                break
            reward = calculateReward(next_state)
            highestReward = max(highestReward, reward[0])
            next_state = list(map(int, next_state))
            next_state = np.reshape(next_state, [1, 8])
            # Store experience
            agent.remember(state, action, reward, next_state)
            if steps_to_update_target_model % 2 == 0:
                agent.replay(batch_size=32, fileObj=logFileObj)
            state = next_state
            total_reward += reward[0]

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        logFileObj.write(f'Total Reward in the Episode {episode+1}:- {total_reward}, Maximum reward in the Episode {episode+1}:- {highestReward}\n')
        env.stopSim()
        time.sleep(2)
    print(f'Highest Reward {highestReward}')
    print("saving model")
    agent.target_model.save_weights('dqn_model.h5')
    logFileObj.close()


if __name__ == '__main__':
    main()