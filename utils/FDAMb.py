import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import copy


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class ReplayMemory(object):
    """
    Create the empty memory buffer with a key-value structure, where key is the vector and value is a tuple (logits, access_count).
    """

    def __init__(self, size):
        self.memory = {}  # {key: (logits, access_count)}
        self.size = size
        self.entry_order = []  # Keep track of entry order for FIFO
        self.memoryindex = 0

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        """
        Store new keys and logits in memory. If memory exceeds the size limit, evict entries based on combined FIFO and LRU strategy.
        """

        dimension = 196608  # Assuming this is the flattened size of key

        key_bytes = keys.reshape(dimension).tobytes()

        if key_bytes in self.memory:
            # If key is already in memory, just update the logits and increment access_count
            logits, access_count, index,protection_count = self.memory[key_bytes]
            self.memory[key_bytes] = (logits, access_count + 1, index, protection_count)
        else:
            # If memory is full, evict the least accessed and oldest entry
            if len(self.memory) >= self.size:
                self.evict_least_used()

            # Add new entry with access_count initialized to 1
            self.memory[key_bytes] = (logits, 1, self.memoryindex,10)
            self.entry_order.append(key_bytes)  # Track insertion order
            self.memoryindex += 1

    def evict_least_used(self):
        """
        Evict the least used and oldest entry in the memory based on access count and FIFO.
        """
        # Find the entry with the lowest access_count and earliest insertion
        min_access_count = float('inf')
        eviction_key = None
        eviction_key_index = None

        for key in self.entry_order:
            _, access_count, index, protection_count = self.memory[key]
            # if protection_count > 0:
            #     self.memory[key] = (_, access_count, protection_count - 1, index)
            #     continue
            if access_count < min_access_count:
                min_access_count = access_count
                eviction_key = key
                eviction_key_index = index


        # Remove the selected key from memory and entry_order
        if eviction_key:
            self.memory.pop(eviction_key)
            self.entry_order.remove(eviction_key)
            print("out",eviction_key_index)


    def _prepare_batch(self, sample):
        """
        Prepare the ensemble prediction from a batch of logits.
        """
        ensemble_prediction = sample[0]

        for logit in sample:
            ensemble_prediction = ensemble_prediction + logit

        ensemble_prediction = ensemble_prediction - sample[0]
        ensemble_prediction = ensemble_prediction / len(sample)
        return torch.FloatTensor(ensemble_prediction)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach, and increments access count of retrieved entries.
        """
        samples = []

        dimension = 196608

        for key in self.memory.keys():
            logits, access_count, index, protection_count = self.memory[key]
            self.memory[key] = (logits, access_count, index, protection_count-1)


        keys = keys.reshape(len(keys), dimension)
        total_keys = len(self.memory.keys())
        self.all_keys = np.frombuffer(
            np.asarray(list(self.memory.keys())), dtype=np.float32).reshape(total_keys, dimension)



        for key in keys:
            similarity_scores = np.dot(self.all_keys, key.T)
            K_neighbour_keys = self.all_keys[np.argpartition(
                similarity_scores, -k)[-k:]]

            neighbours = []
            for nkey in K_neighbour_keys:
                key_bytes = nkey.tobytes()
                logits, access_count, index, protection_count = self.memory[key_bytes]
                # Increment access count for retrieved neighbour
                self.memory[key_bytes] = (logits, access_count + 1, index, protection_count)
                neighbours.append(logits)
                print("index",index,"count",access_count)

            batch = self._prepare_batch(neighbours)
            samples.append(batch)

        return torch.stack(samples)
