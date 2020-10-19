import sys
sys.path.append("..")

from PPO import PPO, Memory
from PIL import Image
from client import Environment
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    ############## Hyperparameters ##############
    env = Environment()

    state_dim = 5+2+3
    action_dim = 15
    
    n_episodes = 9000       # num of episodes to run
    max_timesteps = 300     # max timesteps in one episode
    
    # filename and directory to load model from
    # filename = "PPO_continuous_" +env_name+ ".pth"
    # directory = "./preTrained/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    # ppo.policy_old.load_state_dict(torch.load(directory+filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.new_reset('default')
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.new_step(action, 'default')
            ep_reward += reward
            if done:
                break
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0

if __name__ == '__main__':
    main()
