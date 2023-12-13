import random
import numpy as np

## 다시 하기
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
    
# MC 에서 변경된 점은 update_table 함수가조금 바뀜.
# MC 에서는 update_table 함수가 에이전트가 경험한 history 전체를 인자로 받았지만,
# 이제는 트랜지션(transition)을 인풋으로 받음. 트랜지션은 상태 전이 1번을 뜻함.
# 상태 s에서 a를 해서 보상 r을 받고 상태 s'에 도달했다면 (s,a,r,s')이 한 트랜지션임.
# TD 학습은 샘플 하나만 생기면 바로 업데이트 할 수 있어 history 대신 트랜지션 사용하여 업데이트.
# 업데이트 식도 SARSA 업데이트 식을 그대로 적용함.

class QAgent():

    def __init__(self):
        self.q_table = np.zeros((5, 7, 4)) # q벨류를 저장하는 변수. 모두 0으로 초기화.
        self.eps = 0.9

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
    
    def update_table(self, transition):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        s, a, r, s_prime = transition
        x,y = s
        next_x, next_y = s_prime
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # SARSA 업데이트 식을 이용
        self.q_table[x,y,a] = self.q_table[x,y,a] + 0.1 * (r + self.q_table[next_x,next_y,a_prime] - self.q_table[x,y,a])

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

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000):  # Train for a total of 1,000 episodes
        done = False

        s = env.reset()

        while not done:  # Continue until the end of the episode
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()
    agent.show_table()  # Display the results after learning

if __name__ == '__main__':
    main()