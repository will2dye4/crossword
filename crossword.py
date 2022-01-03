#!/usr/bin/env python3

from collections import defaultdict
from functools import lru_cache
from multiprocessing import cpu_count, Process, Queue
from typing import Optional
import os
import queue
import select
import sys
import time


class Trie:

    class Node:
        def __init__(self, text: str = '', is_terminal: bool = False) -> None:
            self.is_terminal = is_terminal
            self.text = text
            self.children = {}

        def __repr__(self) -> str:
            return str(self)

        def __str__(self) -> str:
            return f'{self.__class__.__name__}({self.text!r}, {self.is_terminal}, {dict(self.children)})'

        @property
        def all_child_words(self) -> set[str]:
            words = set()
            nodes = [self]
            while nodes:
                node = nodes.pop()
                if node.is_terminal:
                    words.add(node.text)
                for child in node.children.values():
                    nodes.append(child)
            return words

        def add_child(self, character: str, text: str, is_terminal: bool = False) -> 'Trie.Node':
            if character in self.children:
                node = self.children[character]
                assert text == node.text
                node.is_terminal = is_terminal
            else:
                node = Trie.Node(text, is_terminal)
                self.children[character] = node
            return node

    def __init__(self) -> None:
        self.root = Trie.Node()

    def __contains__(self, word: str) -> bool:
        return self.has_word(word)

    def __iadd__(self, word: str) -> 'Trie':
        self.add_word(word)
        return self

    @property
    @lru_cache(maxsize=1)
    def all_words(self) -> set[str]:
        return self.root.all_child_words

    @lru_cache(maxsize=128)
    def all_words_with_prefix(self, prefix: str) -> set[str]:
        node = self.root
        for character in prefix:
            if character in node.children:
                node = node.children[character]
            else:
                return set()
        return node.all_child_words

    @lru_cache(maxsize=128)
    def has_prefix(self, prefix: str) -> bool:
        node = self.root
        for character in prefix:
            if character in node.children:
                node = node.children[character]
            else:
                return False
        return True

    def add_word(self, word: str) -> None:
        node = self.root
        for index, character in enumerate(word):
            is_terminal = (index == len(word) - 1)
            if character in node.children:
                next_node = node.children[character]
                if is_terminal:
                    next_node.is_terminal = True
            else:
                text = word[:index+1]
                next_node = node.add_child(character, text, is_terminal)
            node = next_node

    def has_word(self, word: str) -> bool:
        node = self.root
        for character in word:
            if character in node.children:
                node = node.children[character]
            else:
                return False
        return node.is_terminal


def make_tries() -> dict[int, Trie]:
    #start = time.time()
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordlist.txt')
    with open(filepath) as f:
        lines = f.readlines()
    tries = defaultdict(Trie)
    word_count = 0
    for line in lines:
        if not line.startswith('#'):
            line = line.strip().upper()
            if line:
                tries[len(line)].add_word(line)
                word_count += 1
    #end = time.time()
    #print(f'Initialized crossword tries with {word_count:,} words in {end - start:0.2f} seconds')
    return tries


class CrosswordProcess(Process):

    def __init__(self, i: int, puzzle_q: Queue, attempts_q: Queue, generator: 'CrosswordGenerator', **kwargs) -> None:
        super().__init__()
        self.i = i
        self.puzzle_q = puzzle_q
        self.attempts_q = attempts_q
        self.generator = generator
        self.kwargs = kwargs
        self.attempts = 0

    @property
    def row_count(self) -> int:
        return self.generator.row_count

    @property
    def column_count(self) -> int:
        return self.generator.column_count

    @property
    def tries(self) -> dict[int, Trie]:
        return self.generator.tries

    def run(self, row_candidates: Optional[list[str]] = None, prefix: Optional[list[str]] = None) -> Optional[list[str]]:
        if prefix is None:
            prefix = []
        if row_candidates is None:
            row_candidates = list(self.tries[self.column_count].all_words - set(prefix))

        if len(prefix) == self.row_count:
            return prefix

        puzzle = None
        while puzzle is None and row_candidates:
            self.attempts += 1
            if self.attempts % 1_000_000 == 0:
                try:
                    self.attempts_q.put((self.i, self.attempts), block=False)
                except queue.Full:
                    print(f'{self.name} - queue full (count = {self.attempts})')
            next_row = row_candidates.pop()
            if self.generator.is_valid_puzzle_prefix(prefix + [next_row]):
                puzzle = self.run(prefix=prefix + [next_row])

        if puzzle:
            self.puzzle_q.put(puzzle)
        return puzzle


class CrosswordGenerator:

    def __init__(self, row_count: int, column_count: int) -> None:
        self.attempts = 0
        self.row_count = row_count
        self.column_count = column_count
        self.tries = make_tries()
        assert column_count > 0 and column_count in self.tries and len(self.tries[column_count].all_words) >= row_count
        assert row_count > 0 and row_count in self.tries and len(self.tries[row_count].all_words) >= column_count

    def _generate(self, row_candidates: Optional[list[str]] = None,
                  prefix: Optional[list[str]] = None) -> Optional[list[str]]:
        if prefix is None:
            prefix = []
        if row_candidates is None:
            row_candidates = list(self.tries[self.column_count].all_words - set(prefix))

        if len(prefix) == self.row_count:
            return prefix

        puzzle = None
        while puzzle is None and row_candidates:
            self.attempts += 1
            next_row = row_candidates.pop()
            if self.is_valid_puzzle_prefix(prefix + [next_row]):
                puzzle = self._generate(prefix=prefix + [next_row])
        return puzzle

    def generate(self) -> Optional[list[str]]:
        if self.row_count > 5 and self.column_count > 5:
            return self.generate_parallel()

        start = time.time()
        self.attempts = 0
        print(f'Generating a {self.row_count}x{self.column_count} puzzle...')
        puzzle = self._generate()
        end = time.time()
        print(f'Total attempts: {self.attempts:,} ({end - start:0.2f} seconds)')
        return puzzle

    def generate_parallel(self) -> Optional[list[str]]:
        start = time.time()
        num_cpus = cpu_count()
        self.attempts = 0
        print(f'Generating a {self.row_count}x{self.column_count} puzzle...')

        attempts = [0] * num_cpus
        attempts_q = Queue()
        all_row_candidates = list(self.tries[self.column_count].all_words)
        candidates_per_process = len(all_row_candidates) // num_cpus
        processes = []
        puzzle_q = Queue()
        for i in range(num_cpus):
            start_index = i * candidates_per_process
            end_index = len(all_row_candidates) if i == num_cpus - 1 else start_index + candidates_per_process
            row_candidates = all_row_candidates[start_index:end_index]
            process = CrosswordProcess(i, puzzle_q, attempts_q, self, args=(row_candidates,))
            process.start()
            processes.append(process)

        print(f'Started {num_cpus} child processes; waiting for result...')

        puzzle = None
        while not puzzle:
            ready_qs, _, _ = select.select([attempts_q._reader, puzzle_q._reader], [], [])
            if any(fd == puzzle_q._reader for fd in ready_qs):
                puzzle = puzzle_q.get()
            else:
                index, count = attempts_q.get()
                attempts[index] = count

        end = time.time()
        print(f'Total attempts: ~{sum(attempts):,} ({end - start:0.2f} seconds)')

        for process in processes:
            process.terminate()
            process.join()

        return puzzle

    def is_valid_puzzle_prefix(self, puzzle_prefix: list[str]) -> bool:
        # Assumes a rectangular grid with no blank cells
        for column_index in range(self.column_count):
            prefix = ''.join(puzzle_prefix[row_index][column_index] for row_index in range(len(puzzle_prefix)))
            if not self.tries[self.row_count].has_prefix(prefix):
                return False
        return True


def main() -> None:
    if len(sys.argv) < 3:
        rows = 5
        columns = 5
    else:
        rows = int(sys.argv[1])
        columns = int(sys.argv[2])
    generator = CrosswordGenerator(row_count=rows, column_count=columns)
    puzzle = generator.generate()
    print()
    if puzzle:
        for word in puzzle:
            print(' '.join(word))
    else:
        print('Failed to generate puzzle!')


if __name__ == '__main__':
    main()
