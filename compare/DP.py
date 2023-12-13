import numpy as np
import random
import argparse

class DP_Gridworld():
    def __init__(self):
        self.reward = [[-1] * 4 for _ in range(4)]
        self.reward[3][3] = 0
        self.possible_actions = [0,1,2,3] # 상 하 좌 우
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 좌표로 나타낸 행동
        self.transition_probability = 1
        self.all_state = []

        for x in range(4):
            for y in range(4):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = self.actions[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state

    def check_boundary(self,state):
        state[0] = (0 if state[0] < 0 else 3 if state[0] > 3 else state[0])
        state[1] = (0 if state[1] < 0 else 3 if state[1] > 3 else state[1])
        return state
    
class Policy_Iteration():
    def __init__(self,policy_type,env):
        self.env = env
        self.value_table = [[0.0] * 4 for _ in range(4)]
        if policy_type == 'random':
            self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * 4 for _ in range(4)]
            self.policy_table[3][3] = []
        self.discount_factor = 1

    def policy_evaluation(self):
        #다음 가치함수 초기화
        next_value_table = [[0.00] * 4 for _ in range(4)]

        #모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태의 가치 함수 = 0
            if state == [3,3]:
                next_value_table[state[0]][state[1]]=value
                continue

            #벨만 방정식 코드
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state,action)
                reward = self.env.get_reward(state,action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))#transition 확률은 1로

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        if state == [3,3]:
            return 0/0
        return self.policy_table[state[0]][state[1]]
    
    # 가치 함수의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]],2)

def DP_main(policy_type,ne):
    env = DP_Gridworld()
    ne = int(ne)
    pi = Policy_Iteration(policy_type,env)

    val_list = []
    
    for k in range(ne):
        pi.policy_evaluation()

        if ne == 100:
            val_list.append(pi.get_value((0,0)))
        elif ne == 1000:
            if k % 10 == 0:
                val_list.append(pi.get_value((0,0)))
        else:
            if k % 100 == 0:
                val_list.append(pi.get_value((0,0)))
    
    return val_list        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='이 프로그램의 설명(그 외 기타등등 아무거나)')
    #parser.add_argument('--ns', help='N-step')
    parser.add_argument('--policy_type', help='policy_type')
    parser.add_argument('--ne', help='Number of Episodes')
    args = parser.parse_args()

    DP_main(args.policy_type,args.ne)
