import gym
import gym_super_mario_bros
import torch
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from random import random
import torch.nn.functional as F
from gym.spaces import Box
import copy

# --------- Replay Buffer ---------

class ReplayBuffer:
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --------- Mario Preprocessing Wrapper ---------

class MarioWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu", stack_size=4, skip_frames=4):
        super().__init__(env)
        self.device = device
        self.stack_size = stack_size
        self.skip_frames = skip_frames
        self.frames = deque(maxlen=stack_size)
        self.action_space = COMPLEX_MOVEMENT  # Using complex movements
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(stack_size, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        processed = self._preprocess(obs)
        self.frames = deque([processed] * self.stack_size, maxlen=self.stack_size)
        return self._get_tensor()

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        processed = self._preprocess(obs)
        self.frames.append(processed)
        return self._get_tensor(), total_reward, done, info

    def _preprocess(self, frame):
    # Check if the frame is in RGB or another format
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB
            pass  # The frame is already RGB
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA (sometimes present in certain environments)
            frame = frame[:, :, :3]  # Discard the alpha channel
        elif len(frame.shape) == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)  # Convert grayscale to RGB by stacking it

        # Resize and normalize
        resized = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        return normalized  # shape: (84, 84)


    def _get_tensor(self):
        stack = np.stack(self.frames, axis=0)  # (4, 84, 84)
        tensor = torch.tensor(stack, dtype=torch.float32, device=self.device)
        return tensor.unsqueeze(0)  # shape: (1, 4, 84, 84)


# --------- Q-Network ---------

class QNet(torch.nn.Module):
    def __init__(self, n_actions, hidden=512):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        x = x.squeeze(1)
        #print(x.shape)  # Debugging: check input shape
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q



# --------- DQN with Training Loop ---------

def choose_action(state, policy_net, epsilon, n_actions):
    if random() < epsilon:
        return torch.tensor([[np.random.choice(range(n_actions))]])
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            return q_values.max(1)[1].view(1, 1)

def train_dqn(env, policy_net, target_net, replay_buffer, optimizer, gamma=0.99, batch_size=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=100000, num_episodes=1000):
    epsilon = epsilon_start
    total_timesteps = 0
    prev_score = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        policy_checkpoint = {
            'model_state': policy_net.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }

        while not done:
            total_timesteps += 1

            # Choose action (epsilon-greedy)
            action = choose_action(state, policy_net, epsilon, 12)

            # Take action and observe next state
            next_state, reward, done, info = env.step(action.item())
            replay_buffer.push(state, action.item(), reward, next_state, done)

            # Sample from the replay buffer and perform one step of optimization
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Compute Q targets
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    max_next_q_values = next_q_values.max(1)[0]
                    q_targets = rewards + gamma * max_next_q_values * (1 - dones)

                # Get Q values for the current states and actions
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute loss (MSE)
                loss = F.mse_loss(q_values, q_targets)

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update epsilon for epsilon-greedy policy
            epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

            total_reward += reward
            state = next_state

        if total_reward*1.2 < prev_score:
            print(f"Episode {episode} - Total Reward: {total_reward} (Worse than previous)")
            policy_net.load_state_dict(policy_checkpoint['model_state'])
            optimizer.load_state_dict(policy_checkpoint['optimizer_state'])
        else:
            print(f"Episode {episode} - Total Reward: {total_reward} (Better than previous)")
            prev_score = total_reward
        
        # Save the model every 100 episodes 
        if episode % 100 == 0:
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'target_model_state_dict': target_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{episode}.pth')
        # Update target network periodically
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        #print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")


# --------- Initialize the Environment and Networks ---------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Super Mario environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # Use complex actions
    env = MarioWrapper(env, device="cuda", stack_size=4, skip_frames=4)

    # Initialize Q-network and target network
    n_actions = 12
    policy_net = QNet(n_actions).to(device)
    target_net = QNet(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Set up the optimizer
    optimizer = optim.Adam(policy_net.parameters())

    # Initialize ReplayBuffer
    replay_buffer = ReplayBuffer(capacity=10000, device=device)

    # --------- Train the DQN ---------
    try:
        train_dqn(env, policy_net, target_net, replay_buffer, optimizer, gamma=0.99, batch_size=32, num_episodes=3000)
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'target_model_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'target_model_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'checkpoint.pth')