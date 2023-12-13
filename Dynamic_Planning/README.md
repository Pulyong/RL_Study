# Dynamic Planning
Dynamic Planning은 MDP를 알 때 사용할 수 있는 방법입니다. Grid Worl를 GUI로 구현하여 장애물과 목표 지점이 있는 상황에서 목표지점까지 장애물을 피해 최단 거리로 이동하는 Task를 강화학습의 하나인 Dynamic Planning으로 해결합니다.

### Policy Iteration
Policy Iteration은 Policy Evaluation과 Policy Improvement를 번갈아 수행하여 수렴할 때 까지 반복하는 방법론 입니다.  
Policy Evaluation에는 Bellman Expectation Equation을 사용하며 Policy Improvement에는 Greedy Policy Improvement를 이용했습니다.

이론상으로는 Policy Improvement 후에 Policy Evaluation을 state value가 수렴할 때 까지 무한번 반복해야 하나 목표가 최적의 Policy를 찾는 것이기 때문에 6번정도 반복하고 Improvement를 수행해도 여러번 Evaluation과 Improvement를 반복하면 최적의 Policy를 도출 할 수 있습니다.

<img width="494" alt="image" src="https://github.com/Pulyong/RL_Study/assets/76218918/c0fef453-7e32-4f73-b195-86a1c47eb457">

### Value Iteration
Value Iteration은 Bellman Optimality Equation을 사용하여 Optimal Policy를 찾는 방법론 입니다. Value Iteration은 Policy Iteration과 다르게 단계가 나뉘어져 있지 않고 state value가 수렴할 때 까지 반복하여 state value가 큰 action을 greedy하게 수행하면 Optimal policy가 됩니다.

<img width="496" alt="image" src="https://github.com/Pulyong/RL_Study/assets/76218918/57a47ec0-5961-40bf-b6f0-437cefb8574a">