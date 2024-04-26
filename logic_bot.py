from mine_board import MineBoard
import json
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

    def export_game_state(self):

        def get_cell_state(cell):
            state = [0] * 11
            if cell in self.revealed:
                state[self.clues[cell]] = 1
            elif self.board.has_mine(*cell):
                state[10] = 1
            else:
                state[9] = 1
            return state

        return [[get_cell_state((row, col)) for col in range(self.board.width)] for row in range(self.board.height)]

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

    def play_board(self, get_board=False):
        triggered_mines = 0
        game_states = []

        while True:
            if get_board:
                game_states.append(self.export_game_state())

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

        return len(self.revealed), triggered_mines, game_states


def test_bot(width, height, iters):
    print(f"Board Settings: {width}x{height} {iters} iters")
    for i in range(1, 41):
        wins = 0
        avg_triggered_mines = 0
        for _ in range(iters):
            mb = MineBoard(width, height, num_bombs=i)
            bot = SimpleLogicBot(mb)

            num_opened, triggered_mines = bot.play_board()

            avg_triggered_mines += triggered_mines

            if num_opened == width * height - i and triggered_mines == 0:
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

    json.dump(game_states, open(path, "w"), indent=4)

# TODO test this bot once more with the clue update on triggerd mine change


def main():
    collect_data(9, 9, 10, "test_data_gen.json", num_bombs=10)


if __name__ == "__main__":
    main()
