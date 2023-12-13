import random
import numpy as np

class GridWorld():
    def __init__(self):
        # 초기 좌표
        self.x = 0
        self.y = 0

    # action을 받아서 보상을 정해줌
    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()

        #움직일 때마다 -1 so 최단경로로 가겠다
        reward = -1
        done = self.is_done()

        return (self.x, self.y), reward, done
    
    def move_right(self):
        self.y += 1
        if self.y > 3:
            self.y = 3
    
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0

    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
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
    
class Agent():
    # 4방향 random action uniform
    def __init__(self):
        pass

    def select_action(self):
        coin = random.random()

        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3

        return action
    
def main():
    env = GridWorld()
    agent = Agent()

    data = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001 # MC에서의 alpha
    
    for k in range(50000):
        done = False
        history = []

        while not done:
            action = agent.select_action()
            (x, y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        # cum reward 구하기
        cum_reward = 0 # Gt

        for transition in history[::-1]:
            x,y,reward = transition
            data[x][y] = data[x][y] + alpha*(cum_reward - data[x][y]) # V(st) + alpha(Gt - V(st))
            cum_reward = reward + gamma*cum_reward # Gt = Rt+1 + gamma(Gt+1)

    for row in data:
        print(row)

if __name__ == '__main__':
    main()