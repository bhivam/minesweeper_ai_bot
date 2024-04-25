from dataclasses import dataclass
from random import random, shuffle


class MineBoard:

    def __init__(self, width: int, height: int, p=None, num_bombs=None):
        assert height > 0 and width > 0
        assert p == None or num_bombs == None
        if p == None:
            assert num_bombs != None
        if num_bombs == None:
            assert p != None

        self.width = width
        self.height = height

        self.board = [[dict() for i in range(width)] for j in range(height)]

        for row in self.board:
            for cell in row:
                cell["has_mine"] = False

        if p != None:  # use probability
            assert isinstance(p, float) and p > 0 and p < 1

            for row in self.board:
                for cell in row:
                    if random() < p:
                        cell["has_mine"] = True

        if num_bombs != None:
            assert isinstance(num_bombs, int) and num_bombs > 0

            board_indexes = list(range(width * height))
            shuffle(board_indexes)
            bomb_locations = board_indexes[:num_bombs]

            for loc in bomb_locations:
                i = loc // width
                j = loc % width

                self.board[i][j]["has_mine"] = True

    def remove_mine(self, i, j):
        assert i < self.height and i >= 0
        assert j < self.width and j >= 0
        self.board[i][j]["has_mine"] = False

    def get_neighbors(self, i, j, radius=1):
        assert i < self.height and i >= 0
        assert j < self.width and j >= 0
        assert radius > 0

        neighbors = []
        for row in range(i - radius, i + radius + 1):
            for col in range(j - radius, j + radius + 1):
                if (
                    (row == i and col == j)
                    or row < 0
                    or row >= self.width
                    or col < 0
                    or col >= self.height
                ):
                    continue
                neighbors.append((row, col))

        return neighbors

    def has_mine(self, row, col):
        return self.board[row][col]["has_mine"]

    def __str__(self):
        board_str = ""

        for row in self.board:
            for cell in row:
                board_str += "B" if cell["has_mine"] else "x"
                board_str += "  "
            board_str += "\n\n"

        return board_str


def main():
    mb = MineBoard(10, 15, num_bombs=1)
    print(mb)


if __name__ == "__main__":
    main()
