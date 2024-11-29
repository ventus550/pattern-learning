import random
import numpy as np
from typing import Iterable, List


class Automaton:
    def __init__(
        self, transitions: List[List[int]], final_states: List[int] = [], tokens=None
    ):
        self.state = 0
        self.transitions = transitions
        self.final_states = final_states or [len(transitions)]
        self.tokens = tokens or len(transitions[0])

    @property
    def generator(self):
        while self.state not in self.final_states:
            transition = random.randint(0, self.tokens - 1)
            self.state = self.transitions[self.state][transition]
            yield transition
        self.reset()

    def next(self, transition):
        self.state = self.transitions[self.state][transition]
        return self

    def reset(self):
        self.state = 0
        return self


class PatternMatchingAutomaton(Automaton):
    def __init__(self, pattern: Iterable[int], tokens=None):
        states = len(pattern)
        transitions_per_state = len(np.unique(list(pattern)))
        transitions = np.zeros((states, transitions_per_state), dtype=int)
        for state, transition in enumerate(pattern):
            transitions[state, transition] = state + 1
        super().__init__(transitions, tokens=tokens)
