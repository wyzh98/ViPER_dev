import numpy as np
import torch
from collections import deque
import random


buffer_names = ['node_inputs', 'node_padding_mask', 'edge_mask', 'current_index', 'current_edge',
                'edge_padding_mask', 'action', 'reward', 'done', 'all_agent_indices', 'next_node_inputs',
                'next_node_padding_mask', 'next_edge_mask', 'next_current_index', 'next_current_edge',
                'next_edge_padding_mask', 'all_agent_next_indices', 'next_all_agent_next_indices',
                'state_node_inputs', 'state_node_padding_mask', 'state_edge_mask', 'next_state_node_inputs',
                'next_state_node_padding_mask', 'next_state_edge_mask']

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffers = {
            buffer_name: deque(maxlen=buffer_size)
            for buffer_name in buffer_names
        }

    def add(self, data_dict):
        for buffer_name, value in data_dict.items():
            if buffer_name not in self.buffers:
                raise ValueError(f"Buffer '{buffer_name}' does not exist.")
            self.buffers[buffer_name] += value

    def sample(self, device):
        if len(self) < self.batch_size:
            raise ValueError("Not enough data to sample from all buffers.")

        indices = random.sample(range(len(self)), self.batch_size)
        sampled_data = {
            buffer_name: torch.stack([self.buffers[buffer_name][i] for i in indices]).to(device)
            for buffer_name in self.buffers
        }
        return sampled_data

    def __len__(self):
        return len(next(iter(self.buffers.values())))
