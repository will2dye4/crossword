from collections import defaultdict
from functools import lru_cache
from itertools import permutations
from math import factorial
from multiprocessing import cpu_count, Process, Queue
from typing import Optional
import os
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
        if self.row_count > 4 and self.column_count > 4:
            return self.generate_parallel()

        start = time.time()
        self.attempts = 0
        print(f'Generating a {self.row_count}x{self.column_count} puzzle...')
        puzzle = self._generate()
        end = time.time()
        if puzzle:
            print(f'Total attempts: {self.attempts:,} ({end - start:0.2f} seconds)')
        else:
            print('Failed to generate puzzle!')
        return puzzle

    def _generate_parallel(self, q: Queue, row_candidates: Optional[list[str]] = None,
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
                puzzle = self._generate_parallel(q, prefix=prefix + [next_row])

        if puzzle:
            q.put(puzzle)
        return puzzle

    def generate_parallel(self) -> Optional[list[str]]:
        start = time.time()
        self.attempts = 0
        print(f'Generating a {self.row_count}x{self.column_count} puzzle...')

        num_cpus = cpu_count()
        all_row_candidates = list(self.tries[self.column_count].all_words)
        candidates_per_process = len(all_row_candidates) // num_cpus
        processes = []
        q = Queue()
        for i in range(num_cpus):
            start_index = i * candidates_per_process
            end_index = len(all_row_candidates) if i == num_cpus - 1 else start_index + candidates_per_process
            row_candidates = all_row_candidates[start_index:end_index]
            process = Process(target=self._generate_parallel, args=(q, row_candidates))
            process.start()
            processes.append(process)

        print(f'Started {num_cpus} child processes; waiting for result...')
        puzzle = q.get()
        end = time.time()
        print(f'Generated puzzle in {end - start:0.2f} seconds')

        for process in processes:
            process.terminate()
            process.join()

        return puzzle

    def is_valid_puzzle_prefix(self, puzzle_prefix: list[str]) -> bool:
        # Assumes a rectangular grid with no blank cells
        for column_index in range(self.column_count):
            prefix = ''.join(puzzle_prefix[row_index][column_index] for row_index in range(len(puzzle_prefix)))
            if not self.tries[self.row_count].all_words_with_prefix(prefix):
                return False
        return True

    def generate_old(self) -> Optional[list[str]]:
        start = time.time()
        self.attempts = 0
        row_words = self.tries[self.column_count].all_words
        num_permutations = factorial(len(row_words)) // factorial(len(row_words) - self.row_count)
        print(f'Generating a {self.row_count}x{self.column_count} puzzle '
              f'(considering {num_permutations:,} permutations)...')

        for permutation in permutations(row_words, r=self.row_count):
            self.attempts += 1
            if self.is_valid(permutation):
                end = time.time()
                print(f'Total attempts: {self.attempts:,} ({end - start:0.2f} seconds)')
                return list(permutation)

        print('Failed to generate puzzle!')
        return None

    def is_valid(self, words: tuple[str, ...]) -> bool:
        # Assumes a rectangular grid with no blank cells
        top_word = words[0]
        for index in range(len(top_word)):
            maybe_word = ''
            for word in words:
                maybe_word += word[index]
            if maybe_word not in self.tries[len(maybe_word)].all_words:
                return False
        return True


def main() -> None:
    if len(sys.argv) < 3:
        rows = 4
        columns = 4
    else:
        rows = int(sys.argv[1])
        columns = int(sys.argv[2])
    generator = CrosswordGenerator(row_count=rows, column_count=columns)
    puzzle = generator.generate()
    print()
    for word in puzzle:
        print(' '.join(word))


if __name__ == '__main__':
    main()
