import pickle
import sys
from mine_board import MineBoard
from tqdm import tqdm
import statsmodels.stats.api as sms


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

    def get_cell_state_net1(self, cell):
        row, col = cell
        if cell in self.revealed:
            return int(self.clues[cell])
        elif row < 0 or row >= self.board.width or col < 0 or col >= self.board.height:
            return 10  # off board

        elif self.board.has_mine(*cell):
            return 11  # has mine
        else:
            return 9  # unrevealed

    def get_cell_state_net2(self, cell, edge=True):
        if cell in self.revealed:
            return int(self.clues[cell])
        elif self.board.has_mine(*cell) and edge:
            return 10  # has mine
        elif not self.board.has_mine(*cell) and edge:
            return 11
        else:
            return 9  # unrevealed

    def get_example_net1(self, cell, radius=2):
        return [
            [
                self.get_cell_state_net1((i, j))
                for i in range(cell[0] - radius, cell[0] + radius + 1)
            ]
            for j in range(cell[1] - radius, cell[1] + radius + 1)
        ]

    def get_examples_net1(self, radius=2):
        examples = set()

        # get all edge tiles
        for cell in self.revealed:
            for ncell in self.board.get_neighbors(*cell):
                if ncell not in self.revealed:
                    examples.add(self.get_example_net1(ncell, radius=radius))

        return examples

    def get_example_net2(self):
        example = [
            [
                self.get_cell_state_net2((i, j), edge=False)
                for j in range(self.board.width)
            ]
            for i in range(self.board.height)
        ]

        # remove all mines not on edge
        for i in range(self.board.height):
            for j in range(self.board.width):
                if (i, j) not in self.revealed:
                    for ncell in self.board.get_neighbors(i, j):
                        if ncell in self.revealed:
                            example[i][j] = self.get_cell_state_net2((i, j))
                            break
        return example

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

    def play_board(self, get_data=None):
        triggered_mines = 0
        game_states = []
        revealed_before_bomb = 0
        correct_predictions = 0

        while True:
            if get_data == "net1":
                game_states.extend(self.get_examples_net1())
            elif get_data == "net2":
                game_states.append(self.get_example_net2())

            row, col = None, None
            if len(self.safe) > 0:
                row, col = self.safe.pop()
            elif len(self.remaining) > 0:
                if self.board.start_cell in self.remaining:
                    row, col = self.board.start_cell
                    self.remaining.remove((row, col))
                else:
                    row, col = self.remaining.pop()
            else:
                break

            if self.board.has_mine(row, col):
                if triggered_mines == 0:
                    revealed_before_bomb = len(self.revealed)

                triggered_mines += 1
                self.board.remove_mine(row, col)

                neighbors = self.board.get_neighbors(row, col)
                for neighbor in neighbors:
                    if neighbor in self.clues.keys():
                        self.clues[neighbor] = max(self.clues[neighbor] - 1, 0)
            else:
                correct_predictions += 1

            self.revealed.add((row, col))
            self.clues[(row, col)] = self.board.board[row][col]["clue"]

            while True:
                if not self.make_inference():
                    break

        if revealed_before_bomb == 0:
            revealed_before_bomb = len(self.revealed)

        return (
            correct_predictions,
            revealed_before_bomb,
            triggered_mines,
            list(game_states),
        )


def test_bot(width, height, iters, min_bombs, max_bombs):
    print(f"Board Settings: {width}x{height} {iters} iters")
    for num_bombs in range(min_bombs, max_bombs + 1):
        wins = []
        mts = []
        revealed = []
        correct_rates = []
        for _ in tqdm(range(iters), leave=False):
            mb = MineBoard(width, height, num_bombs=num_bombs)
            bot = SimpleLogicBot(mb)

            cpr, revealed_i, triggered_mines, _ = bot.play_board()

            mts.append(triggered_mines)

            if triggered_mines == 0:
                wins.append(1)
            else:
                wins.append(0)

            correct_rates.append(cpr / (height * width + triggered_mines - num_bombs))
            revealed.append(revealed_i)

        wins_conf = sms.DescrStatsW(wins).tconfint_mean()
        revealed_conf = sms.DescrStatsW(revealed).tconfint_mean()
        cr_conf = sms.DescrStatsW(correct_rates).tconfint_mean()
        mt_conf = sms.DescrStatsW(mts).tconfint_mean()

        format_ci = (
            lambda ci, is_percent: f"{(100 if is_percent else 1)*(ci[0]+ci[1])/2:.2f}±{(100 if is_percent else 1)*abs(ci[0]-(ci[0]+ci[1])/2):.2f}"
        )

        print(
            f"Bombs: {num_bombs} |Average MT: {format_ci(mt_conf, False)}| Average Revealed: {format_ci(revealed_conf, False)} | WR: {format_ci(wins_conf, True)}%| Average CPR: {format_ci(cr_conf, True)}%"
        )


def collect_mixed_data(iters, path, format="net1"):
    examples = []
    for i in tqdm(range(iters)):
        p = (i % 25 + 6) / 100
        mb = MineBoard(9, 9, p=p)
        bot = SimpleLogicBot(mb)
        _, _, _, examples_i = bot.play_board(get_data=format)
        examples.extend(examples_i)

    pickle.dump(examples, open(path, "wb"))


def collect_data(width, height, iters, path, p=None, num_bombs=None, format="net1"):
    assert format == "net1" or format == "net2"
    game_states = []

    for _ in tqdm(range(iters)):
        mb = MineBoard(width, height, p, num_bombs)
        bot = SimpleLogicBot(mb)

        _, _, _, game_states_i = bot.play_board(get_data=format)

        game_states.extend(game_states_i)

    pickle.dump(game_states, open(path, "wb"))


def main():
    argv = sys.argv
    argc = len(argv)

    if argv[1] == "benchmark":
        if argc != 7:
            print(
                "Usage: python3 logic_bot.py benchmark <board height> <board width> <min # of bombs> <max # of bombs> <iters>"
            )
            sys.exit(1)
        _, _, height, width, min_bombs, max_bombs, iters = argv

        height, width, min_bombs, max_bombs, iters = (
            int(height),
            int(width),
            int(min_bombs),
            int(max_bombs),
            int(iters),
        )

        print(
            f"Model: Logic Bot | HxW: {height}x{width} | Bomb Range: {min_bombs}-{max_bombs} | {iters} Iterations"
        )
        test_bot(height, width, iters, min_bombs, max_bombs)
    elif argv[1] == "datagen":
        if argc != 5:
            print("usage: python3 logic_bot.py datagen <iters> <data path> <format>")
            sys.exit(1)

        _, _, iters, data_path, format = argv
        assert format == "net1" or format == "net2"

        iters = int(iters)

        collect_mixed_data(iters, data_path, format=format)

    elif argv[1] == "old_datagen":
        if argc != 8:
            print(
                "Usage: python3 logic_bot.py old_datagen <board height> <board width> <# of bombs> <iters> <data path> <format>"
            )
            sys.exit(1)

        _, _, height, width, num_bombs, iters, data_path, format = argv
        height, width, num_bombs, iters = (
            int(height),
            int(width),
            int(num_bombs),
            int(iters),
        )

        collect_data(
            height, width, iters, data_path, num_bombs=num_bombs, format=format
        )
    else:
        print("Usage: python3 logic_bot.py [benchmark old_datagen]")


if __name__ == "__main__":
    main()
