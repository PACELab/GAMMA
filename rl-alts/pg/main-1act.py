import sys
import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.autograd import Variable

sys.path.append("..")

import matplotlib.pyplot as plt
from client import Environment

PLOT_FIG = False

# constants
GAMMA = 0.9
NUM_RESOURCES = 5
LR = 3e-4
HIDDEN_SIZE = 40
NUM_EPS = 48000
NUM_STEPS = 300
ID = 'default'

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=LR):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear_act = nn.Linear(hidden_size, int(num_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x_act = self.linear_act(x)
        x_act = F.softmax(x_act, dim=1)

        # probability list
        return x_act
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        
        highest_prob_action = np.random.choice(int(self.num_actions), p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])

        return highest_prob_action, log_prob

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    # normalize discounted rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
        # policy_gradient.append(1.0/(log_prob) * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    # policy_gradient.backward()
    policy_gradient.backward(retain_graph=True)
    policy_network.optimizer.step()

def main():
    # environment for getting states and peforming actions
    env = Environment()

    # policy network
    policy_net = PolicyNetwork(NUM_RESOURCES+2+3, NUM_RESOURCES*3, HIDDEN_SIZE)
    
    max_episode_num = NUM_EPS
    max_steps = NUM_STEPS
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    avg_rewards = []

    action_step = 10
    available_actions = [0, action_step, -action_step]

    for episode in range(max_episode_num):
        # initilization
        state = env.new_reset(ID)
        # state = env.transfer_reset()
        log_probs = []
        rewards = []

        """
        if episode < 290:
            max_steps = episode+10
        else:
            max_steps = NUM_STEPS
        """

        for steps in range(max_steps):
            curr_arrival_rate = state['curr_arrival_rate']
            cpu_limit = state['cpu_limit']
            mem_limit = state['mem_limit']
            llc_limit = state['llc_limit']
            io_limit = state['io_limit']
            net_limit = state['net_limit']
            curr_cpu_util = state['curr_cpu_util']
            curr_mem_util = state['curr_mem_util']
            curr_llc_util = state['curr_llc_util']
            curr_io_util = state['curr_io_util']
            curr_net_util = state['curr_net_util']
            slo_retainment = state['slo_retainment']
            rate_ratio = state['rate_ratio']
            percentages = state['percentages']

            if episode == max_episode_num-1:
                print("EP:", episode, " | Step:", steps)
                print("Update - Current SLO Retainment:", slo_retainment)
                print("Update - Current Util:", str(curr_cpu_util)+'/'+str(cpu_limit), str(curr_mem_util)+'/'+str(mem_limit), str(curr_llc_util)+'/'+str(llc_limit), str(curr_io_util)+'/'+str(io_limit), str(curr_net_util)+'/'+str(net_limit))

            state_vector = np.asarray([curr_cpu_util/cpu_limit,curr_mem_util/mem_limit,curr_llc_util/llc_limit,curr_io_util/io_limit,curr_net_util/net_limit,slo_retainment,rate_ratio,percentages[0],percentages[1],percentages[2]], dtype = float)
            action, log_prob = policy_net.get_action(state_vector)

            cpu_action = 0
            if action < 3:
                cpu_action = available_actions[action]
            mem_action = 0
            if action >= 3 and action < 6:
                mem_action = available_actions[action-3]
            llc_action = 0
            if action >= 6 and action < 9:
                llc_action = available_actions[action-6]
            io_action = 0
            if action >= 9 and action < 12:
                io_action = available_actions[action-9]
            net_action = 0
            if action >= 12:
                net_action = available_actions[action-12]

            if episode == max_episode_num-1:
                print("Update - Actions to take:", cpu_action, mem_action, llc_action, io_action, net_action)

            new_state, reward, done = env.new_step(cpu_action, mem_action, llc_action, io_action, net_action, ID)
            
            log_probs.append(log_prob)
            rewards.append(reward)

            if episode == max_episode_num-1:
                print("Update - Reward got:", reward, '\n')

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break
            
            state = new_state
        update_policy(policy_net, rewards, log_probs)
        numsteps.append(steps)
        avg_numsteps.append(np.mean(numsteps[-10:]))
        all_rewards.append(np.sum(rewards))
        avg_rewards.append(np.mean(all_rewards[-10:]))
        if episode % 1 == 0:
            sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
        if PLOT_FIG:
            if episode % 1000 ==0 and episode != 0:
                plt.plot(all_rewards, color='darkorange')  # total rewards in an iteration or episode
                plt.plot(avg_rewards, color='b')  # (moving avg) rewards
                plt.xlabel('Episodes')
                plt.savefig('ep'+str(episode)+'.png')
    if PLOT_FIG:
        # plt.plot(numsteps)     # (realtime) time to recover
        # plt.plot(avg_numsteps) # (moving avg) average time to recover
        plt.plot(all_rewards, color='darkorange')  # total rewards in an iteration or episode
        plt.plot(avg_rewards, color='b')  # (moving avg) rewards
        plt.xlabel('Episodes')
        plt.savefig('final.png')
        
        # write to file
        f = open("avg_reward.txt", "w")
        for reward in avg_rewards:
            f.write(str(reward)+"\n")
        f.close()
        f = open("total_reward.txt", "w")
        for reward in all_rewards:
            f.write(str(reward)+"\n")
        f.close()

if __name__ == '__main__':
    main()
