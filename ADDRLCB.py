import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import torchvision.transforms as T
from PIL import Image

# Hyperparameters
NUM_ACTIONS = 18
MAX_DURATION = 20
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 1000000
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE = 10000
LEARNING_RATE_DQN = 0.00025
LEARNING_RATE_BANDIT = 0.0001
NUM_EPISODES = 200
FRAME_STACK = 4
FRAME_SIZE = (84, 84)

# Frame Preprocessing
def preprocess_frame(frame):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize(FRAME_SIZE),
        T.ToTensor()
    ])
    return transform(frame).squeeze(0)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, duration, reward, next_state, done):
        self.memory.append((state, action, duration, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN with Bandit Duration Estimation
class BanditDQN(nn.Module):
    def __init__(self, num_actions, max_duration):
        super(BanditDQN, self).__init__()
        self.conv1 = nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_dqn = nn.Linear(64 * 7 * 7, 1024)
        self.fc2_dqn = nn.Linear(1024, num_actions)
        self.fc_bandit = nn.Linear(64 * 7 * 7, max_duration)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        q_values = F.relu(self.fc1_dqn(x))
        q_values = self.fc2_dqn(q_values)
        duration_logits = self.fc_bandit(x)
        duration_probs = F.softmax(duration_logits, dim=-1)
        return q_values, duration_probs

# Atari Environment Wrapper (Gym v0.26+)
class AtariEnvWrapper:
    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.action_space = self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        state = preprocess_frame(obs)
        return torch.stack([state] * FRAME_STACK, dim=0)

    def step(self, action, duration):
        total_reward = 0
        for _ in range(duration):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        next_state = preprocess_frame(obs)
        return next_state, total_reward, done

    def close(self):
        self.env.close()

# Training Loop
def train_bandit_dqn(env_name):
    env = AtariEnvWrapper(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = BanditDQN(env.action_space, MAX_DURATION).to(device)
    target_net = BanditDQN(env.action_space, MAX_DURATION).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer_dqn = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_DQN)
    optimizer_bandit = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE_BANDIT)
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    
    steps_done = 0
    episode_rewards = []

    for episode in range(NUM_EPISODES):
        state = env.reset().to(device)
        episode_reward = 0
        done = False
        
        while not done:
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
            if random.random() < epsilon:
                action = random.randrange(env.action_space)
            else:
                with torch.no_grad():
                    q_values, _ = policy_net(state.unsqueeze(0))
                    action = q_values.argmax().item()
            
            with torch.no_grad():
                _, duration_probs = policy_net(state.unsqueeze(0))
                duration = torch.multinomial(duration_probs, 1).item() + 1
            
            next_frame, reward, done = env.step(action, duration)
            episode_reward += reward
            
            next_state = torch.cat([state[1:], preprocess_frame(next_frame).unsqueeze(0).to(device)], dim=0)
            memory.push(state, action, duration, reward, next_state, done)
            
            state = next_state
            steps_done += duration
            
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = list(zip(*transitions))
                
                states = torch.stack(batch[0]).to(device)
                actions = torch.tensor(batch[1], device=device)
                durations = torch.tensor(batch[2], device=device)
                rewards = torch.tensor(batch[3], device=device, dtype=torch.float32)
                next_states = torch.stack(batch[4]).to(device)
                dones = torch.tensor(batch[5], device=device, dtype=torch.float32)
                
                q_values, _ = policy_net(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_values, _ = target_net(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    targets = rewards + (1 - dones) * GAMMA * max_next_q
                
                dqn_loss = F.mse_loss(q_values, targets)
                optimizer_dqn.zero_grad()
                dqn_loss.backward()
                optimizer_dqn.step()
                
                _, duration_probs = policy_net(states)
                log_probs = torch.log(duration_probs.gather(1, (durations - 1).unsqueeze(1)).squeeze(1))
                
                with torch.no_grad():
                    next_q_values, _ = policy_net(next_states)
                    next_actions = next_q_values.argmax(dim=1)
                    next_q_selected = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    bandit_rewards = next_q_selected - q_values
                
                bandit_loss = -(bandit_rewards * log_probs).mean()
                optimizer_bandit.zero_grad()
                bandit_loss.backward()
                optimizer_bandit.step()
            
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    env.close()
    return episode_rewards

# Auto-select valid Seaquest environment
if __name__ == "__main__":
    available_envs = gym.envs.registry.keys()
    seaquest_envs = [env for env in available_envs if "Seaquest" in env]

    if not seaquest_envs:
        raise ValueError("No Seaquest environment found. Run: pip install gym[atari,accept-rom-license]")
    
    selected_env = seaquest_envs[0]
    print(f"Using environment: {selected_env}")
    train_bandit_dqn(selected_env)
