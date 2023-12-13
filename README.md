# RL_Study
Reinforcement Learning 공부를 정리하기 위해서 만든 레포입니다.
각 폴더 내에서 자세한 내용을 볼 수 있으며 간단한 내용은 다음과 같습니다.

## Dynamic Planning
MDP를 알 때 사용하는 강화학습의 방법론 입니다. Policy Iteration과 Value Iteration이 있습니다. MDP를 알 경우에만 사용할 수 있으므로 정확한 값을 구할 수 있지만 현실세계에 적용할 때에는 제약이 있습니다.

## Model Free Value Prediction
MDP를 모를 때 state value를 예측할 수 있는 방법론 입니다. MDP를 모르기 때문에 여러번의 에피소드를 반복하며 state value를 업데이트 합니다. Monte Carlo(MC), N_step_TD 방법이 존재합니다. MC는 에피소드가 종료하고 나서 value를 업데이트 하는 반면 TD는 에피소드 내에서 한번의 step이 진행되면 바로 업데이트를 진행하는 방식입니다.  
  
MC는 Bias가 작은 대신 Variance가 크고 TD는 반대로 Bias가 큰 대신 Variance가 작습니다. TD에서 N의 수가 커질 수록 MC와 비슷해 집니다.

## Model Free Control
Value Prediction이 아니라 실제 optimal policy를 구하기 위한 방법론 입니다. state value가 아닌 action value를 사용합니다. MC 기반의 MonteCarlo Control과 TD기반의 SARSA,Q-Learning이 존재합니다.
SARSA는 다음 Action을 epsilon greedy에 의해 선택하고 업데이트 하지만 Q-Learning에서는 다음 Action을 max value를 뽑아 업데이트 합니다.

## DQN
DQN은 Q-Learning 기반의 딥러닝 방법론으로 Agent를 Deep Learning 구조로 대체한 방법론 입니다. Deep Learning을 통해 State를 Input으로 넣으면 Action value가 output으로 나오는 형태입니다.