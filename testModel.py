import pickle
import time

from exec_environment import Simulation, calculateReward, mapStates, DQNAgent
from exec_environment import getState

import numpy as np

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

class TestAgent:
    def __init__(self, q_values):
        self.q_values = q_values

    def select_action(self, state):
        # Exploit: choose the action with the highest learned Q-value
        a = np.argmax(self.q_values[state, :])
        if(self.q_values[state, a]==0):
            return np.random.randint(0, len(self.q_values[0]))
        return a
import time

def testWthRandom(qtablePath, testEpisodes, testSteps):
    global states
    thresholdReward = 54
    d={}
    testAgent = TestAgent('')
    with open(qtablePath, 'rb') as f:
        testAgent.q_values = pickle.load(f)
    start_time = time.time()
    diff = float('inf')
    for episode in range(testEpisodes):
        print(f'Running Test episode: {episode + 1}')
        env = Simulation()
        positions = env.getObjectsPositions()
        state = getState(positions)
        highestReward = float('-inf')
        while state==-1:
            env.stopSim()
            time.sleep(2)
            env = Simulation()
            positions = env.getObjectsPositions()
            state = getState(positions)

        for _ in range(testSteps):
            #directionIndex = testAgent.select_action(mappedStates[state])
            directionIndex = np.random.randint(0, 4)
            env.action(direction=env.directions[directionIndex])
            positions = env.getObjectsPositions()
            new_state = getState(positions)
            if new_state == -1:
                break
            reward = calculateReward(new_state)
            if(reward[0]>=thresholdReward):
                diff = min(diff, time.time() - start_time)
                d[episode] = 1
                # print("diff random", diff)
            highestReward = max(reward[0], highestReward)
            # print(f"Action Taken: {env.directions[directionIndex]}, Reward on taking this action: {reward[0]} Uniformity: {reward[1]} Distibution of uniformity: {reward[2]}")
            state = new_state
        print(f'Highest Reward in episode {episode+1} is {highestReward}')

        env.stopSim()
        time.sleep(1)
    print(f"Total Time: {time.time()-start_time} sec")
    print(f"Perfectly mixed first at {diff}")
    cnt = 0
    for k in d:
        if d[k]==1:
            cnt+=1
    print(f'Number of successful mixes over ten trails: {cnt}')


def testWithQtable(qtablePath, testEpisodes, testSteps):
    episodes = 30
    logFile = 'logtest.txt'
    cn=0
    logFileObj = open(logFile, 'w')
    steps_to_update_target_model = 0
    highestReward = float('-inf')
    # qAgent = QLearningAgent(48400, 4, 0.3, 0.9, 0.1)
    agent = DQNAgent(48400, 4)
    agent.target_model(np.reshape([1,2,3,4,5,6,7,8], [1, 8]))
    agent.target_model.load_weights(qtablePath)
    total_reward = 0
    global states
    mappedStates = mapStates(states)
    for episode in range(testEpisodes):
        print(f'Running episode: {episode + 1}')
        env = Simulation()
        positions = env.getObjectsPositions()
        state = getState(positions)
        total_reward = 0
        while state == -1:
            env.stopSim()
            time.sleep(2)
            env = Simulation()
            positions = env.getObjectsPositions()
            state = getState(positions)
        highestReward = float('-inf')
        state = np.reshape(list(map(int, state)), [1, 8])
        for _ in range(testSteps):
            steps_to_update_target_model += 1
            # here
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
            state = next_state
            total_reward += reward[0]
            if(reward[0]>=52):
                cn+=1

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
        logFileObj.write(
            f'Total Reward in the Episode {episode + 1}:- {total_reward}, Maximum reward in the Episode {episode + 1}:- {highestReward}\n')
        env.stopSim()
        time.sleep(2)
    print(f'Highest Reward {highestReward}')
    print("saving model")
    logFileObj.close()
    print(f"Successful Runs: {cn}")

import time

start = time.time()
testWithQtable('dqn_model.h5', 100, 80)


print("Training:- Take actions based on Q Table:-")
print(f"Total Time:- {time.time()-start}")

