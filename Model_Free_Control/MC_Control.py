import random
import numpy as np

# action을 받아서 보상을 정해줌

class GridWorld:
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, a):
    # Define action mapping: 0 for left, 1 for up, 2 for right, 3 for down
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()

        reward = -1  # The reward is always -1
        done = self.is_done()
        
        next_state = (self.x, self.y)
        
        return next_state, reward, done


    def move_left(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1

    def move_right(self):
        if self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        elif self.y == 6:
            pass
        else:
            self.y += 1

    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1

    def is_done(self):
        if self.x == 4 and self.y == 6:  # If the goal point (4,6) is reached, it's done
            return True
        else:
            return False
    
    # x,y좌표 return
    def get_state(self):
        return (self.x, self.y)
    
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x,self.y)
    
# 에이전트에 해당하는 객체를 정의함.
# 에이전트 객체는 내부에 q(s, a)의 값을 저장하기 위해 테이블을 갖고 있음.
# 이 테이블은 실제 에이전트가 액션을 선택할 때 사용됨.
# select_action 함수를 통해 상태 s를 인풋으로 받아 s에서 알맞은 액션을 입실론 그리디 방식으로 선택
# epsilon 값은 0.9에서 시작하여 0.1까지 선형적으로 줄어듬.
# update_table 함수는 실제로 테이블 값을 업데이트 해주는 함수임.
# 한 에피소드에 해당하는 데이터를 받아 MC 방법으로 테이블 값을 업데이트 함.
# anneal_eps 함수는 epsilon 값을 점차 조금씩 줄여주기 위해 필요한 함수임.
# show_table 함수는 학습이 끝난 후에 상태별로 q(s,a)의 값이 가장 큰 액션을 뽑아서 보여주는 함수임.

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # q벨류를 저장하는 변수. 모두 0으로 초기화.
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self, s):
    # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
        return action
    
    def update_table(self, history):
    # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x,y = s
            # 몬테 카를로 방식을 이용하여 업데이트.
            self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (cum_reward - self.q_table[x,y,a])
            cum_reward = cum_reward + r

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)

# 에이전트와 환경을 만들고, 하나의 에피소드가 끝날 때까지 history라는 변수에 상태 전이 과정을
# 모두 저장해 두었다가, 에피소드가 끝난 순간 해당 변수를 이용해 에이전트 내부의 q 테이블업데이트.
# 그리고 epsilon의 값을조금씩 줄여주며, 이렇게총 1천 번의 에피소드 동안 학습하고 최종 결과 출력.

def main():
    env = GridWorld()
    agent = QAgent()

    for episode in range(1000):  # Train for a total of 1,000 episodes
        done = False
        history = []

        s = env.reset()

        while not done:  # Continue until the end of the episode
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime

        agent.update_table(history)  # Update the agent using the history
        agent.anneal_eps()
    agent.show_table()  # Display the results after learning

if __name__ == '__main__':
    main()
