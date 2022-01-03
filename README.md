# crossword &mdash; word grid generator

This is a Python script for generating grids of words. It allows you to specify
the number of rows and columns you want and attempts to generate a grid of words
with those dimensions.

## Usage

### Prerequisites

The script assumes you are running Python 3.9 or later.

### Arguments

The script expects two arguments: the number of rows and the number of columns
for the word grid. If you only specify one or neither, the default is 5x5.

### Examples

```
$ ./crossword.py
Generating a 5x5 puzzle...
Total attempts: 5,511,270 (5.33 seconds)

Z O H A N
B L O T S
A D B O T
R E I N A
S N E E R
```

```
$ ./crossword.py 4 10
Generating a 4x10 puzzle...
Started 10 child processes; waiting for result...
Total attempts: ~128,000,000 (29.98 seconds)

L O V E T O B I T S
O V E R E M O T E S
C E N T R A L I S T
I N T E R S E C T S
```

## About

### Algorithm

The script attempts to find a list of words that form a valid word grid of the
desired size, i.e., each row and column of the grid is a valid word according to
the included word list (`wordlist.txt`). The algorithm uses
[tries](https://en.wikipedia.org/wiki/Trie) in order to check if a prefix or word
is valid efficiently. It iterates over the valid words of the given length in 
random order and uses the tries to check subsequent random words to see if they 
form valid columns.

### Performance

This algorithm described above seems to perform well enough for small puzzles
(up to about 5x5). If you request a puzzle that's too large, the script will
launch several child processes (one for each CPU core that your machine has)
in an attempt to generate the puzzle faster. When running in parallel, however,
the accuracy of the statistics keeping track of the total attempts is sacrificed.
If one of the child processes happens to find a valid puzzle before the other
child processes have started up, the script will report zero total attempts.

For puzzles above about 6x6, the script may take a very long time (or possibly
forever) to run. The number of permutations of valid words increases substantially
as the grid size increases (especially the number of rows), and the algorithm may still have to check billions
of billions (of billions of billions...) of permutations depending on the size
you requested.

### Word List

The file `wordlist.txt` contains all the words considered valid by the script.
Add words to this file, or remove them from the file, to adjust the script's
word list.

### Crossword Puzzles

As of this writing (January 2022), the script generates only grids of words with
no blank cells. Future work may include modifying the script to be capable of
generating actual crossword puzzles, where each row or column may contain multiple
words separated by blank spaces.
