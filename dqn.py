import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter 



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)




class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size= action_size
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.model = DQN(screen_height, screen_width, n_actions).to(device)
        self.target = DQN(screen_height, screen_width, n_actions).to(device)
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
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


if __name__ == '__main__':
    num_episodes = 50
    env = gym.make('CartPole-v0').unwrapped
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
    
    