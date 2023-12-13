# CartPole Game using DQN
DQN은 Q-Learning을 바탕으로 Agent를 Deep Learning으로 구현한 모델입니다. Input은 state가 들어가며 output은 Action vector입니다.

## Behavior Q & Target Q
DQN은 두개의 Q Network를 사용합니다. Behavior Q에서 여러번 에피소드를 진행하여 파라미터를 업데이트하고 일정 iteration마다 한번씩 Target Q에 Behavior Q의 파라미터를 복사합니다. Target Q는 ground truth로 사용되며 Behavior Q는 Q-learning 방식으로 action을 선택하여 reward와 함께 ground truth와의 차이를 loss로 사용합니다.

## Replay Buffer
DQN에서는 Replay Buffer가 사용됩니다. 에피소드 내에서 경험이 이루어 질 때 바로바로 경험을 바탕으로 Behavior Q를 train 하는 것이 아니라, buffer에 넣어놓고 buffer가 어느정도 찼을 때 train을 진행합니다. Replay Buffer는 직관적으로 과거의 경험을 버리는 것이 아니라 재사용 할 수 있게 합니다.
