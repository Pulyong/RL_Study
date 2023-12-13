import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from DP import *
from MC import *
from N_Step_TD import *

def main(policy_type,ne):
    
    result = dict()
    mc_list = []
    os_list = []
    ts_mc_list = []
    fs_mc_list = []
    for num_episode in ne:
        DP_for_optimal_value = np.array(DP_main(policy_type,num_episode))[-1]
        
        result[f'MC_{num_episode}'] = np.abs(np.array(MC_main(policy_type,num_episode)) - DP_for_optimal_value)
        result[f'1S_TD_{num_episode}'] = np.abs(np.array(TD_main(1,policy_type,num_episode)) - DP_for_optimal_value)
        result[f'3S_TD_{num_episode}'] = np.abs(np.array(TD_main(3,policy_type,num_episode)) - DP_for_optimal_value)
        
        
    
    plt.figure(figsize=(20, 7))
    
    
    # episode 100
    x = list(range(1, 101))
    plt.subplot(311)
    plt.plot(x,result['MC_100'],label='MC')
    plt.plot(x,result['1S_TD_100'],label='1S_TD')
    plt.plot(x,result['3S_TD_100'],label='3S_TD')
    
    
    plt.ylabel('abs error')
    plt.title('Episode 100')

    # episode 1000
    x = list(range(1, 1001))
    plt.subplot(312)
    plt.plot(x,result['MC_1000'],label='MC')
    plt.plot(x,result['1S_TD_1000'],label='1S_TD')
    plt.plot(x,result['3S_TD_1000'],label='3S_TD')
    
    
    plt.ylabel('abs error')
    plt.title('Episode 1000')
    
    # episode 10000
    x = list(range(1, 10001))
    plt.subplot(313)
    plt.plot(x,result['MC_10000'],label='MC')
    plt.plot(x,result['1S_TD_10000'],label='1S_TD')
    plt.plot(x,result['3S_TD_10000'],label='3S_TD')
    
    
    plt.ylabel('abs error')
    plt.title('Episode 10000')

    # 그래프를 표시
    plt.legend()
    plt.show()


    plt.figure(figsize=(20, 7))
    x = list(range(1, 10001))
    plt.plot(x,result['MC_10000'],label='MC')
    plt.plot(x,result['1S_TD_10000'],label='1S_TD')
    plt.plot(x,result['3S_TD_10000'],label='3S_TD')
    plt.ylabel('abs error')
    plt.xlabel('num episode')
    plt.title('Episode 10000')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='이 프로그램의 설명(그 외 기타등등 아무거나)')
    #parser.add_argument('--ns', help='N-step')
    parser.add_argument('--policy_type', help='policy_type')
    parser.add_argument('--ne', help='Number of Episodes')
    args = parser.parse_args()

    ne = list(map(int,args.ne.split(',')))

    main(args.policy_type,ne)