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
import matplotlib.pyplot as plt

BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 0.999 
MEMORY_SIZE = 50000
NUM_EPISODE =1000
TARGET_UPDATE=2
GAMMA = 0.99 # 시간할인율

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            next_state))
                            
    def sample(self, BATCH_SIZE):
        
        return random.sample(self.memory, BATCH_SIZE)
               
    
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
        #print('DQN')
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
            return torch.LongTensor([[random.randrange(self.action_size)]])
        else:
            with torch.no_grad():
                return self.model(state).max(1)[1].view(1, 1)
    
    def optimize_model(self,done):
        if len(self.memory) < BATCH_SIZE:
            return
        if self.epsilon > EPS_END :
            self.epsilon *= EPS_DECAY
        #print('training')
        batch = self.memory.sample(BATCH_SIZE)
        
        state, action, reward, next_state = zip(*batch)
 
        state = torch.cat(state)        
        action = torch.cat(action)
        reward = torch.cat(reward)
        next_state = torch.cat(next_state)
        
        #여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고
        # 이에 대한 Q값을 gather 메서드로 모아온다
        current_q = self.model(state).gather(1, action) 
        max_next_q = torch.zeros(BATCH_SIZE)
        max_next_q = self.model(next_state).detach().max(1)[0]
        #print(self.model(next_state))
        #print(self.model(next_state).max(1)[0].detach().max(1)[0])
        expected_q = reward + (1 - done)*GAMMA * max_next_q
        
        self.model.train()
        
        loss = F.smooth_l1_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            

        
        return loss

if __name__ == '__main__':
    #env=gym.make('CartPole-v0')
    
    #env=gym.make('LunarLander-v2')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    steps = 0
    agent = DQNAgent(state_size, action_size)
    writer = SummaryWriter('run')
    score_history = []
    for e in range(NUM_EPISODE):
        #env.render()
        state = env.reset()
        state = torch.FloatTensor([state])
        done = False
        score =0
        loss = 0
        loss_list =[]

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action.item())
            score += reward
            next_state = torch.FloatTensor([next_state])
            agent.memory.push(state,action,reward,next_state)
            
            if agent.memory.__len__() >= MEMORY_SIZE:
                #print('training')
                loss = agent.optimize_model(done)
                loss_list.append(loss.item())
            
            state = next_state
            steps += 1
        l= np.mean(loss_list)    
        writer.add_scalar("Loss", l, e)
        writer.add_scalar("Scores", score, e)
        score_history.append(score)
        print("에피소드:{0} 점수: {1}".format(e, score))

        
	    # Update the target network, copying all weights and biases in DQN
        if e % TARGET_UPDATE == 0:
            agent.target.load_state_dict(agent.model.state_dict())    
    
plt.plot(score_history)
plt.ylabel('score')
plt.show()
    
    