# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# Lib
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import os

# Files
from noise import OrnsteinUhlenbeckActionNoise as OUNoise
from replaybuffer import Buffer
from actorcritic import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.0003
CRITIC_LR = 0.003
MINIBATCH_SIZE = 64
NUM_EPISODES = 9000
NUM_TIMESTEPS = 300
MU = 0
SIGMA = 0.2
CHECKPOINT_DIR = './checkpoints/manipulator/'
BUFFER_SIZE = 100000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
EPSILON = 1.0
EPSILON_DECAY = 1e-6

def obs2state(state_list):
    """Converts observation dictionary to state tensor"""
    #l1 = [val.tolist() for val in list(observation.values())]
    #l2 = []
    #for sublist in l1:
    #    try:
    #        l2.extend(sublist)
    #    except:
    #        l2.append(sublist)
    return torch.FloatTensor(state_list).view(1, -1)

CUDA = True

class DDPG:
    def __init__(self, env):
        self.env = env
        self.stateDim = 5+2+3
        self.actionDim = 3*5
        if CUDA:
            self.actor = Actor(self.env).cuda()
            self.critic = Critic(self.env).cuda()
            self.targetActor = deepcopy(Actor(self.env)).cuda()
            self.targetCritic = deepcopy(Critic(self.env)).cuda()
        else:
            self.actor = Actor(self.env)
            self.critic = Critic(self.env)
            self.targetActor = deepcopy(Actor(self.env))
            self.targetCritic = deepcopy(Critic(self.env))
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.actionDim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.checkpoint_dir = CHECKPOINT_DIR
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []
        self.start = 0
        self.end = NUM_EPISODES

    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):       
        """Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets"""
        if CUDA:
            targetBatch = torch.FloatTensor(rewardBatch).cuda() 
        else:
            targetBatch = torch.FloatTensor(rewardBatch)
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s != True, terminalBatch)))
        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        nextActionBatch.volatile = True
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        
        nonFinalMask = self.discount * nonFinalMask.type(torch.cuda.FloatTensor)
        targetBatch += nonFinalMask * qNext.squeeze().data
        
        return Variable(targetBatch, volatile=False)

    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + \
                                          TAU*orgParam.data)

    def getMaxAction(self, curState):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""
        if CUDA:
            noise = self.epsilon * Variable(torch.FloatTensor(self.noise()), volatile=True).cuda()
        else:
            noise = self.epsilon * Variable(torch.FloatTensor(self.noise()), volatile=True)
        action = self.actor(curState)
        actionNoise = action + noise
        return actionNoise
        
    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        print('Training started...')
        
        action_step = 10
        available_actions = [0, action_step, -action_step]
        all_rewards = []
        avg_rewards = []
        # for each episode 
        for i in range(self.start, self.end):
            state = self.env.reset()
            
            ep_reward = 0
            
            for step in range(NUM_TIMESTEPS):
            # while not time_step.last():
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

                # print each time step only at the last EPISODE
                if episode == NUM_EPISODES-1:
                    print("EP:", episode, " | Step:", step)
                    print("Update - Current SLO Retainment:", slo_retainment)
                    print("Update - Current Util:", str(curr_cpu_util)+'/'+str(cpu_limit), str(curr_mem_util)+'/'+str(mem_limit), str(curr_llc_util)+'/'+str(llc_limit), str(curr_io_util)+'/'+str(io_limit), str(curr_net_util)+'/'+str(net_limit))

                # Get maximizing action
                if CUDA:
                    curState = Variable(obs2state([curr_cpu_util/cpu_limit,curr_mem_util/mem_limit,curr_llc_util/llc_limit,curr_io_util/io_limit,curr_net_util/net_limit,slo_retainment,rate_ratio,percentages[0],percentages[1],percentages[2]]), volatile=True).cuda()
                else:
                    curState = Variable(obs2state([curr_cpu_util/cpu_limit,curr_mem_util/mem_limit,curr_llc_util/llc_limit,curr_io_util/io_limit,curr_net_util/net_limit,slo_retainment,rate_ratio,percentages[0],percentages[1],percentages[2]]), volatile=True) 
                self.actor.eval()     
                action = self.getMaxAction(curState)
                curState.volatile = False
                action.volatile = False
                self.actor.train()
                
                # Step episode
                time_step = self.env.step(action.data)
                if CUDA:
                    nextState = Variable(obs2state(time_step.observation), volatile=True).cuda()
                else:
                    nextState = Variable(obs2state(time_step.observation), volatile=True)
                reward = time_step.reward
                ep_reward += reward
                terminal = time_step.last()
                
                # Update replay bufer
                self.replayBuffer.append((curState, action, nextState, reward, terminal))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = torch.cat(curStateBatch)
                    actionBatch = torch.cat(actionBatch)
                    
                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)
                    
                    # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    print('Critic Loss: {}'.format(criticLoss))
                    criticLoss.backward()
                    self.criticOptim.step()
            
                    # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -torch.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    print('Actor Loss: {}'. format(actorLoss))
                    actorLoss.backward()
                    self.actorOptim.step()
                    
                    # Update Targets                        
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay
                    
            if i % 20 == 0:
                self.save_checkpoint(i)
            self.rewardgraph.append(ep_reward)

    def save_checkpoint(self, episode_num):
        checkpointName = self.checkpoint_dir + 'ep{}.pth.tar'.format(episode_num)
        checkpoint = {
            'episode': episode_num,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'targetActor': self.targetActor.state_dict(),
            'targetCritic': self.targetCritic.state_dict(),
            'actorOpt': self.actorOptim.state_dict(),
            'criticOpt': self.criticOptim.state_dict(),
            'replayBuffer': self.replayBuffer,
            'rewardgraph': self.rewardgraph,
            'epsilon': self.epsilon
            
        } 
        torch.save(checkpoint, checkpointName)
    
    def loadCheckpoint(self, checkpointName):
        if os.path.isfile(checkpointName):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpointName)
            self.start = checkpoint['episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.targetActor.load_state_dict(checkpoint['targetActor'])
            self.targetCritic.load_state_dict(checkpoint['targetCritic'])
            self.actorOptim.load_state_dict(checkpoint['actorOpt'])
            self.criticOptim.load_state_dict(checkpoint['criticOpt'])
            self.replayBuffer = checkpoint['replayBuffer']
            self.rewardgraph = checkpoint['rewardgraph']
            self.epsilon = checkpoint['epsilon']
            print('Checkpoint loaded')
        else:
            raise OSError('Checkpoint not found')
