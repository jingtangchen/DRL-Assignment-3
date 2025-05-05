import gym
import torch
import numpy as np
from train import QNet, MarioWrapper
from collections import deque

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        
        self.device = torch.device('cpu')
        self.model = QNet(12).to(self.device)
        self.model.eval()
        try:
            checkpoint = torch.load('checkpoint.pth', map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print("Error loading checkpoint: ", e)
            
        self.frames = deque(maxlen=4)
        self.flag = False
        self.wrapper = MarioWrapper(None, device=self.device)

    def act(self, observation):
        if self.flag is False:
            self.flag = True
            # Preprocess the observation
            observation = self.wrapper._preprocess(observation)
            self.frames = deque([observation] * 4, maxlen=4)
        else:
            # Preprocess the observation
            observation = self.wrapper._preprocess(observation)
            self.frames.append(observation)

        stack = np.stack(self.frames, axis=0)  # (4, 84, 84)
        tensor = torch.tensor(stack, dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.model(tensor)
            action = q_values.max(1)[1].view(1, 1)
        return action.item()
    
