
import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
import torch
import logging

logger = logging.getLogger("rllib.replay-buffer")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=42):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float(

        ).to(
            device)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(1)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)

        # print(f"states.size()={states.size()}")
        # print(f"actions.size()={actions.size()}")
        # print(f"rewards.size()={rewards.size()}")
        # print(f"done.size()={dones.shape}")
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

#  TODO: merge the 2 implementations together (test for speed etc.)
# class ReplayBuffer(object):
#     def __init__(self, size):
#         """Create Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0
#
#     def __len__(self):
#         return len(self._storage)
#
#     def add(self, obs_t, action, reward, obs_tp1, done):
#         data = (obs_t, action, reward, obs_tp1, done)
#
#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize
#
#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#             dones.append(done)
#         return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
#
#     def sample(self, batch_size):
#         """Sample a batch of experiences.
#
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         """
#         idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes)



# class PrioritizedReplayBuffer(ReplayBuffer):
#     def __init__(self, size, alpha):
#         """Create Prioritized Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedReplayBuffer, self).__init__(size)
#         assert alpha >= 0
#         self._alpha = alpha
#
#         it_capacity = 1
#         while it_capacity < size:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#
#     def add(self, *args, **kwargs):
#         """See ReplayBuffer.store_effect"""
#         idx = self._next_idx
#         super().add(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha
#
#     def _sample_proportional(self, batch_size):
#         res = []
#         p_total = self._it_sum.sum(0, len(self._storage) - 1)
#         every_range_len = p_total / batch_size
#         for i in range(batch_size):
#             mass = random.random() * every_range_len + i * every_range_len
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res
#
#     def sample(self, batch_size, beta):
#         """Sample a batch of experiences.
#
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#
#
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         weights: np.array
#             Array of shape (batch_size,) and dtype np.float32
#             denoting importance weight of each sampled transition
#         idxes: np.array
#             Array of shape (batch_size,) and dtype np.int32
#             idexes in buffer of sampled experiences
#         """
#         assert beta > 0
#
#         idxes = self._sample_proportional(batch_size)
#
#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * len(self._storage)) ** (-beta)
#
#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / max_weight)
#         weights = np.array(weights)
#         encoded_sample = self._encode_sample(idxes)
#         return tuple(list(encoded_sample) + [weights, idxes])
#
#     def update_priorities(self, idxes, priorities):
#         """Update priorities of sampled transitions.
#
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#
#         Parameters
#         ----------
#         idxes: [int]
#             List of idxes of sampled transitions
#         priorities: [float]
#             List of updated priorities corresponding to
#             transitions at the sampled idxes denoted by
#             variable `idxes`.
#         """
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha
#
#             self._max_priority = max(self._max_priority, priority)

#
#
# import threading
#
# import numpy as np
#
#
# class ReplayBuffer:
#     def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
#         """Creates a replay buffer.
#
#         Args:
#             buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
#                 buffer
#             size_in_transitions (int): the size of the buffer, measured in transitions
#             T (int): the time horizon for episodes
#             sample_transitions (function): a function that samples from the replay buffer
#         """
#         self.buffer_shapes = buffer_shapes
#         self.size = size_in_transitions // T
#         self.T = T
#         self.sample_transitions = sample_transitions
#
#         # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
#         self.buffers = {key: np.empty([self.size, *shape])
#                         for key, shape in buffer_shapes.items()}
#
#         # memory management
#         self.current_size = 0
#         self.n_transitions_stored = 0
#
#         self.lock = threading.Lock()
#
#     @property
#     def full(self):
#         with self.lock:
#             return self.current_size == self.size
#
#     def sample(self, batch_size):
#         """Returns a dict {key: array(batch_size x shapes[key])}
#         """
#         buffers = {}
#
#         with self.lock:
#             assert self.current_size > 0
#             for key in self.buffers.keys():
#                 buffers[key] = self.buffers[key][:self.current_size]
#
#         buffers['o_2'] = buffers['o'][:, 1:, :]
#         buffers['ag_2'] = buffers['ag'][:, 1:, :]
#
#         transitions = self.sample_transitions(buffers, batch_size)
#
#         for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
#             assert key in transitions, "key %s missing from transitions" % key
#
#         return transitions
#
#     def store_episode(self, episode_batch):
#         """episode_batch: array(batch_size x (T or T+1) x dim_key)
#         """
#         batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
#         assert np.all(np.array(batch_sizes) == batch_sizes[0])
#         batch_size = batch_sizes[0]
#
#         with self.lock:
#             idxs = self._get_storage_idx(batch_size)
#
#             # load inputs into buffers
#             for key in self.buffers.keys():
#                 self.buffers[key][idxs] = episode_batch[key]
#
#             self.n_transitions_stored += batch_size * self.T
#
#     def get_current_episode_size(self):
#         with self.lock:
#             return self.current_size
#
#     def get_current_size(self):
#         with self.lock:
#             return self.current_size * self.T
#
#     def get_transitions_stored(self):
#         with self.lock:
#             return self.n_transitions_stored
#
#     def clear_buffer(self):
#         with self.lock:
#             self.current_size = 0
#
#     def _get_storage_idx(self, inc=None):
#         inc = inc or 1   # size increment
#         assert inc <= self.size, "Batch committed to replay is too large!"
#         # go consecutively until you hit the end, and then go randomly.
#         if self.current_size+inc <= self.size:
#             idx = np.arange(self.current_size, self.current_size+inc)
#         elif self.current_size < self.size:
#             overflow = inc - (self.size - self.current_size)
#             idx_a = np.arange(self.current_size, self.size)
#             idx_b = np.random.randint(0, self.current_size, overflow)
#             idx = np.concatenate([idx_a, idx_b])
#         else:
#             idx = np.random.randint(0, self.size, inc)
#
#         # update replay size
#         self.current_size = min(self.size, self.current_size+inc)
#
#         if inc == 1:
#             idx = idx[0]
#         return idx
# from collections import deque
# import random
# from utilities import transpose_list
#
#
# class ReplayBuffer:
#     def __init__(self, size):
#         self.size = size
#         self.deque = deque(maxlen=self.size)
#
#     def push(self, transition):
#         """push into the buffer"""
#
#         input_to_buffer = transpose_list(transition)
#
#         for item in input_to_buffer:
#             self.deque.append(item)
#
#     def sample(self, batchsize):
#         """sample from the buffer"""
#         samples = random.sample(self.deque, batchsize)
#
#         # transpose list of list
#         return transpose_list(samples)
#
#     def __len__(self):
#         return len(self.deque)
#
#
#


# class PrioritizedReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, action_size, buffer_size, batch_size, seed, use_prioritized_replay=False):
#         """Initialize a ReplayBuffer object.
#
#         Params
#         ======
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             seed (int): random seed
#         """
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",
#                                                                 "proba"])
#         self.seed = random.seed(seed)
#         self.use_prioritized_replay = use_prioritized_replay
#         if self.use_prioritized_replay:
#             logger.info("Using Prioritized Replay.")
#         self.alpha = 0.8  # 0=pure randomness, 1=pure probabilities
#         self.beta = 0.1
#         self.beta_increment_per_sampling = 0.001
#
#     def add(self, state, action, reward, next_state, done, proba=None):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done, proba)
#         self.memory.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         if self.use_prioritized_replay:
#             self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
#             probas = np.array([m.proba for m in self.memory])**self.alpha
#             probas = probas / np.sum(probas)
#             experiences_ids = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=probas)
#         else:
#             experiences_ids = random.sample(range(len(self.memory)), k=self.batch_size)
#
#         experiences = np.array(self.memory)[experiences_ids]
#
#         states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(
#             device)
#         dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
#             device)
#
#         if self.use_prioritized_replay:
#             weights = (len(self.memory) * probas) ** (-self.beta)
#             weights /= np.max(weights)
#             weights = torch.from_numpy(np.vstack(weights)).float().to(device)
#         else:
#             weights = None
#
#         return (states, actions, rewards, next_states, dones, weights, experiences_ids)
#
#     def update_priorities(self, ids, errors):
#         for exp_id, e in zip(ids, errors):
#             self.memory[exp_id]._replace(proba=abs(e))
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.memory)