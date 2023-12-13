import numpy as np
import random

HEIGHT = 4  # 그리드월드 세로
WIDTH = 4  # 그리드월드 가로
TRANSITION_PROB = 1  # 전이 확률
POSSIBLE_ACTIONS = [0, 1, 2, 3]  # 상, 하, 좌, 우
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 좌표로 나타낸 행동
REWARDS = []

class Env:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH  # Grid World의 너비
        self.height = HEIGHT  # Grid World의 높이
        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[2][2] = 1  # 원에 대한 보상 1
        self.reward[1][2] = -1  # 삼각형에 대한 보상 -1
        self.reward[2][1] = -1  # 삼각형에 대한 보상 -1
        self.all_states = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_states.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1 if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1 if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_states
