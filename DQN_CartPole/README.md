# CartPole Game using DQN
DQN은 Q-Learning을 바탕으로 Agent를 Deep Learning으로 구현한 모델입니다. Input은 state가 들어가며 output은 Action vector입니다.

## Behavior Q & Target Q
DQN은 두개의 Q Network를 사용합니다. Behavior Q에서 여러번 에피소드를 진행하여 파라미터를 업데이트하고 일정 iteration마다 한번씩 Target Q에 Behavior Q의 파라미터를 복사합니다. Target Q는 ground truth로 사용되며 Behavior Q의 Loss와 함께 train에 이용됩니다.

![image](https://github.com/Pulyong/RL_Study/assets/76218918/07bd2540-bfad-45d0-8eeb-d8473819788b)

위 식에서 $y_i$는 Target Q를 이용하여 TD 방식으로 predict한 action value로 ground truth로 이용합니다. Bahavior Q가 predict한 action value는 ground truth와 같아지도록 training이 진행됩니다.

## Replay Buffer
DQN에서는 Replay Buffer가 사용됩니다. 에피소드 내에서 경험이 이루어 질 때 바로바로 경험을 바탕으로 Behavior Q를 train 하는 것이 아니라, buffer에 넣어놓고 buffer가 어느정도 찼을 때 train을 진행합니다. Replay Buffer는 직관적으로 과거의 경험을 버리는 것이 아니라 재사용 할 수 있게 합니다. 

여기에서 구현된 Replay Buffer는 buffer 안에 있는 sample들 중 random으로 batch만큼을 sampling하여 사용합니다. random으로 sampling하는 이유는 buffer에 값이 들어갈 때 결국 time에 dependent해지기 때문에 time independent하게 sample을 뽑아오기 위함입니다. 하지만 생각을 조금 더 해보면 이 방식은 naive 하다는 사실을 알 수 있습니다. 현실을 생각했을 때도 조금 더 중요한 경험이 있고 너무 일상적이라 학습에 별로 도움이 되지 않는 경험들도 있기 때문입니다. 이러한 사실에 입각하여 replay buffer에서 sampling하는 방식 또한 하나의 연구 과제이고 많은 방식이 연구되어있습니다.

## Algorithm
![image](https://github.com/Pulyong/RL_Study/assets/76218918/0a311951-3979-4a81-a33f-3f96016436fa)

알고리즘을 설명하면 다음과 같습니다.

1. 에피소드를 n번 반복한다.
2. 각 에피소드를 시행하면서 Buffer에 state, action, reward, next action을 넣는다.
3. buffer가 일정 수준 (구현에서는 2000) 이상 들어있다면 training을 진행한다.
4. 일정 interval마다 behavior Q에서 target Q로 parameter를 update 시킨다.

training 과정은 다음과 같습니다.

1. behavior Q로 action value를 predict한다.
2. Target Q에 next state를 넣어 next action value를 가져온다.
3. next action value에 gamma를 곱하고 reward를 더해서 ground truth action value를 구한다.
4. behavior Q의 action value와 ground truth와의 smooth l1 loss로 loss를 구한다.
5. behavior Q를 업데이트 한다.
