import random
import numpy as np
from collections import deque

class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.batch = None

    def push(self, data_tuple):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(None)
        self.buffer[-1] = data_tuple

    def sample(self, batch_size, resample=True):
        if resample:
            self.batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*self.batch))
    
    def dump(self, batch_size):
        return map(np.stack, zip(*self.buffer[-batch_size:]))

    def reset(self):
        self.buffer = []
        self.batch = None

    def __len__(self):
        return len(self.buffer)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch = None

    def push(self, data_tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, resample=True):
        if resample:
            self.batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*self.batch))
    
    def dump(self, batch_size):
        if self.position > batch_size:
            self.batch = self.buffer[self.position - batch_size - 1:self.position] + []
        else:
            self.batch = self.buffer[self.position - batch_size - 1:] + self.buffer[:self.position]
        return map(np.stack, zip(*self.batch))

    def reset(self):
        self.buffer = []
        self.position = 0
        self.batch = None

    def __len__(self):
        return len(self.buffer)

class SimpleEpisodicMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch = None

    def push(self, data_tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, resample=True):
        if resample:
            self.batch = random.sample(self.buffer, batch_size)
        return self.batch

    def reset(self):
        self.buffer = []
        self.position = 0
        self.batch = None

    def __len__(self):
        return len(self.buffer)

class EpisodeReplayMemory:
    def __init__(self, capacity, kws, shapes, dtypes):
        self.capacity = capacity
        self.buffer = dict()
        self.kws = kws
        for i in range(len(kws)):
            self.buffer[kws[i]] = np.zeros((self.capacity,) + shapes[i], dtype=dtypes[i])
        self.sep = deque()
        self.curr_start = 0
        self.position = 0
        self.episodes = 0

    def push(self, **kwargs):
        if len(self.sep)>0:
            if self.sep[0][0] == self.position:
                self.sep.popleft()
        for key in kwargs:
            if key == 'done':
                if kwargs[key]:
                    # meet episode end signal
                    if self.position >= self.curr_start:
                        # ensure every episode is continuous
                        self.sep.append((self.curr_start, self.position))
                    self.curr_start = int((self.position + 1) % self.capacity)
            self.buffer[key][self.position] = kwargs[key]

        self.position = int((self.position + 1) % self.capacity)
        self.count = max(self.count, self.position)

    def reset(self):
        self.position = 0
        self.count = 0
        self.sep = deque()
    
    def sample(self, batch_size):
        indices = np.random.choice(self.count, batch_size, replace=False)
        batch = dict()
        for kw in self.kws:
            batch[kw] = self.buffer[kw][indices]

        return [batch[kw] for kw in self.kws]

    def sample_episodes(self, batch_size, kw='obs'):
        assert len(self.sep) >= batch_size
        indices = np.random.choice(len(self.sep), batch_size, replace=False)
        batch = []
        for idx in indices:
            start, end = self.sep[idx]
            batch.append(self.buffer[kw][start:end+1])
        return batch
