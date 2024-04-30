import pickle
import sys
from mine_board import MineBoard
from tqdm import tqdm


class SimpleLogicBot:

    def __init__(self, board: MineBoard):
        self.board = board

        self.remaining = set()
        self.revealed = set()
        self.safe = set()
        self.mine = set()
        self.clues = dict()

        for row in range(board.height):
            for col in range(board.width):
                self.remaining.add((row, col))

    def get_cell_state(self, cell):
        row, col = cell
        if cell in self.revealed:
            return int(self.clues[cell])
        elif row < 0 or row >= self.board.width or col < 0 or col >= self.board.height:
            return 10  # off board

        elif self.board.has_mine(*cell):
            return 11  # has mine
        else:
            return 9  # unrevealed
 
    def get_example(self, cell, radius=2):
        return [
                [
                    self.get_cell_state((i, j))
                    for i in range(cell[0] - radius, cell[0] + radius + 1)
                    ]
                for j in range(cell[1] - radius, cell[1] + radius + 1)
                ]
    
    def get_examples(self, radius=2):
        examples = set()

        # get all edge tiles 
        for cell in self.revealed:
            for ncell in self.board.get_neighbors(*cell):
                if ncell not in self.revealed: 
                    examples.add(self.get_example(ncell, radius=radius))

        return examples


    def make_inference(self):
        made_inference = False
        for crow, ccol in self.clues.keys():
            neighbors = set(self.board.get_neighbors(crow, ccol))
            adj_inf_mine = neighbors.intersection(self.mine)
            adj_inf_safe = neighbors.intersection(self.safe)
            adj_revealed = neighbors.intersection(self.revealed)
            adj_unrevealed = neighbors.difference(adj_revealed)

            if self.clues[(crow, ccol)] - len(adj_inf_mine) == len(adj_unrevealed):
                for cell in adj_unrevealed:
                    made_inference = True
                    self.remaining.remove(cell)
                    self.mine.add(cell)
            elif (len(neighbors) - self.clues[(crow, ccol)]) - (
                len(adj_revealed) + len(adj_inf_safe)
            ) == len(adj_unrevealed):
                for cell in adj_unrevealed:
                    made_inference = True
                    self.remaining.remove(cell)
                    self.safe.add(cell)

        return made_inference

    def play_board(self, get_data=False):
        triggered_mines = 0
        game_states = set()

        while True:
            if get_data:
                game_states.update(self.get_examples())

            row, col = None, None
            if len(self.safe) > 0:
                row, col = self.safe.pop()
            elif len(self.remaining) > 0:
                row, col = self.remaining.pop()
            else:
                break

            if self.board.has_mine(row, col):
                triggered_mines += 1
                self.board.remove_mine(row, col)

                neighbors = self.board.get_neighbors(row, col)
                for neighbor in neighbors:
                    if neighbor in self.clues.keys():
                        self.clues[neighbor] = max(self.clues[neighbor]-1, 0)

            self.revealed.add((row, col))
            self.clues[(row, col)] = self.board.board[row][col]["clue"]

            while True:
                if not self.make_inference():
                    break

        return len(self.revealed), triggered_mines, list(game_states)


def test_bot(width, height, iters, min_bombs, max_bombs):
    print(f"Board Settings: {width}x{height} {iters} iters")
    for i in range(min_bombs, max_bombs+1):
        wins = 0
        avg_triggered_mines = 0
        for _ in range(iters):
            mb = MineBoard(width, height, num_bombs=i)
            bot = SimpleLogicBot(mb)

            _, triggered_mines, _ = bot.play_board()

            avg_triggered_mines += triggered_mines

            if triggered_mines == 0:
                wins += 1

        print(f"{i} bombs | {100*wins/iters:.2f}% WR | {avg_triggered_mines /
              iters:.2f} TM | {100*avg_triggered_mines/iters/i:.2f}% MTR")


def collect_data(width, height, iters, path, p=None, num_bombs=None):
    game_states = []

    for _ in tqdm(range(iters)):
        mb = MineBoard(width, height, p, num_bombs)
        bot = SimpleLogicBot(mb)

        _, _, game_states_i = bot.play_board(True)

        game_states.extend(game_states_i)

    pickle.dump(game_states, open(path, "wb"))


def main():
    argv = sys.argv
    argc = len(argv)

    if argv[1] == "benchmark":
        if argc != 7:
            print(
                "Usage: python3 nn_bot.py visual <board height> <board width> <min # of bombs> <max # of bombs> <iters>"
            )
            sys.exit(1)
        _, _, height, width, min_bombs, max_bombs, iters = argv

        print(
            f"Model: Logic Bot | HxW: {height}x{width} | Bomb Range: {min_bombs}-{max_bombs} | {iters} Iterations"
        )
        test_bot(height, width, iters, min_bombs, max_bombs)
    elif argv[1] == "old_datagen":
        if argc != 7:
            print(
                "Usage: python3 nn_bot.py old_datagen <model path> <board height> <board width> <# of bombs> <iters> <data path>"
            )
            sys.exit(1)

        _, _, height, width, num_bombs, iters, data_path = argv
        height, width, num_bombs, iters = int(height), int(width), int(num_bombs), int(iters)

        collect_data(height, width, iters, data_path, num_bombs=num_bombs)



if __name__ == "__main__":
    main()
