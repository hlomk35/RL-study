import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from collections import deque
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter 


BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 0.999 
MEMORY_SIZE = 1000
NUM_EPISODE =300
TARGET_UPDATE=10

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        #차원 확장해주기. []->[[]]
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append((state, action, reward, next_state))
    
    def sample(self, BATCH_SIZE):
        state, action, reward, next_state = zip(*random.sample(self.memory, BATCH_SIZE))
        return torch.tensor(np.concatenate(state), dtype=torch.float), torch.tensor(action), \
               torch.tensor(reward), torch.tensor(np.concatenate(next_state), dtype=torch.float), 
               
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self,state_size,action_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size))
        
    def forward(self, x):
        print('DQN')
        return self.layers(x)
    
  

class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size= action_size
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.model = DQN(self.state_size, self.action_size)
        self.target = DQN(self.state_size, self.action_size)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(),lr= LEARNING_RATE)
        
        self.epsilon = EPS_START
        self.target
    
    def get_action(self,state):
        if np.random.rand()<=self.epsilon:
            return torch.tensor([[random.randrange(self.action_size)]],dtype=torch.long)
        else:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
    
    def optimize_model(self,done):
        if self.epsilon > EPS_END :
            self.epsilon *= EPS_DECAY
    
        state, action, reward, next_state = self.memory.sample(BATCH_SIZE)
        

        q_values          = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = self.target(next_state).max(1)[0]

        expected_q_value = reward + (1 - done)*LEARNING_RATE*next_q_value
        
        loss = F.smooth_l1_loss(q_values, expected_q_value.unsqueeze(1)) #huber loss
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

if __name__ == '__main__':
    #env=gym.make('CartPole-v0')
    
    env=gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    writer = SummaryWriter('run')

    for e in range(NUM_EPISODE):
        state = env.reset()
        state      = np.concatenate(np.expand_dims(state, 0))
        state = torch.from_numpy(state)
        
        done = False
        score =0
        loss = 0
        loss_list =[]

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action.item())
            agent.memory.push(state,action,reward,next_state)
            
            if agent.memory.__len__() >= MEMORY_SIZE:
                print('training')
                loss = agent.optimize_model(done)
                loss_list.append(loss.item())
            
            state = next_state
        l= np.mean(loss_list)    
        writer.add_scalar("Loss", l, e)
        writer.add_scalar("Scores", score, e)

        
	    # Update the target network, copying all weights and biases in DQN
        if e % TARGET_UPDATE == 0:
            agent.target.load_state_dict(agent.model.state_dict())    
    
    print('done')
    
    