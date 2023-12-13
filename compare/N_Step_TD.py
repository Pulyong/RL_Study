import random
import numpy as np
import argparse

class TD_Gridworld():
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

    def set_state(self,state):
        self.x = state[0]
        self.y = state[1]

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x,self.y)
    
class Agent():

    # 4방향 random action uniform
    def __init__(self,policy_type):
        self.policy_type = policy_type

    def select_action(self):
        if self.policy_type == 'random':
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

    
def TD_main(ns,policy_type,ne):
    
    ns = int(ns)
    ne = int(ne)

    
    gamma = 1
    alpha = 0.01 # TD에서의 alpha

    env = TD_Gridworld()
    agent = Agent(policy_type)
    data = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    val_list = []
    for k in range(ne):
        done = False

        while not done:
            x, y = env.get_state()
            Gt = 0
            for n in range(ns):
                action = agent.select_action()    
                (x_prime,y_prime), reward, done = env.step(action) # 다음 상태
                
                if n == 0:
                    mem_x,mem_y = x_prime,y_prime
                    
                Gt += reward*np.power(gamma,n)

                if done or n == ns-1:
                    Gt += np.power(gamma,n+1)*data[x_prime][y_prime]
                    break
            data[x][y] = data[x][y] + alpha * (Gt - data[x][y])
            env.set_state((mem_x,mem_y))

        env.reset()
        val_list.append(data[0][0])

    return val_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='이 프로그램의 설명(그 외 기타등등 아무거나)')
    parser.add_argument('--ns', help='N-step')
    parser.add_argument('--policy_type', help='policy_type')
    parser.add_argument('--ne', help='Number of Episodes')
    args = parser.parse_args()

    TD_main(args.ns,args.policy_type,args.ne)
    
