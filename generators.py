from dataclasses import dataclass
import numpy as np
import torch
import random

def markov_chain_probability(k, m, n):
    # Transition matrix for Markov chain
    P = np.zeros((m+1, m+1))

    # Fill transition probabilities
    for i in range(m):
        P[i, i] = (k - 1) / k  # Stay in the same state if the character doesn't match
        P[i, i+1] = 1 / k      # Move to the next state if the character matches

    P[m][m] = 1
    return np.linalg.matrix_power(P, n)[0][m]

def binsearch_m(k, n, target_prob=0.5):
    # Binary search for the m that gives a probability closest to target_prob (default 0.5)
    left, right = 1, n
    best_m, best_prob = None, float('inf')

    while left <= right:
        mid = (left + right) // 2
        prob = markov_chain_probability(k, mid, n)

        if prob > target_prob:
            left = mid + 1
        else:
            right = mid - 1

        if abs(prob - target_prob) < abs(best_prob - target_prob):
            best_m, best_prob = mid, prob
    return best_m, best_prob


@dataclass
class SubsequenceDiscriminationData:
    subsequence: list | int = None
    batch_size: int = 64
    length: int = 24
    length_deviation: float = 0.0
    characters: int = 2
    frequency = None

    def __post_init__(self):
        if isinstance(self.subsequence, int) or self.subsequence is None:
            if self.subsequence is None:
                self.subsequence, _ = binsearch_m(self.characters, self.length)
            self.subsequence = self.sample_sequences(1, self.subsequence)[0]
        assert len(self.subsequence) <= self.length
        self.frequency = markov_chain_probability(self.characters, len(self.subsequence), self.length)

    def has_subsequence(self, sequence):
        i = 0
        for char in sequence:
            i += i < len(self.subsequence) and char == self.subsequence[i]
            if i == len(self.subsequence):
                return True
        return False

    def sample_sequences(self, count, length):
        return np.random.randint(0, self.characters, (count, length))

    def __call__(self, *args, **kwds):
        X = self.sample_sequences(self.batch_size, max(len(self.subsequence), round(random.gauss(self.length, self.length_deviation))))
        Y = torch.tensor([self.has_subsequence(x) for x in X]).float()
        return torch.tensor(X), Y