from generate_boards import MineBoard


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

    def play_board(self):
        triggered_mines = 0

        while True:
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
                        self.clues[neighbor] -= 1

            self.revealed.add((row, col))

            neighbors = self.board.get_neighbors(row, col)

            clue_count = 0
            adj_inf_mine = set()
            adj_inf_safe = set()
            adj_unrevealed = set()
            adj_revealed = set()
            for neighbor in neighbors:
                nrow, ncol = neighbor
                if self.board.has_mine(nrow, ncol):
                    clue_count += 1
                if (nrow, ncol) in self.mine:
                    adj_inf_mine.add(neighbor)
                if (nrow, ncol) in self.safe:
                    adj_inf_safe.add(neighbor)
                if (nrow, ncol) not in self.revealed:
                    adj_unrevealed.add(neighbor)
                else:
                    adj_revealed.add(neighbor)

            self.clues[(row, col)] = clue_count

            if clue_count - len(adj_inf_mine) == len(adj_unrevealed):
                for cell in adj_unrevealed:
                    self.remaining.remove(cell)
                    self.mine.add(cell)
            elif (len(neighbors) - clue_count) - (
                len(adj_revealed) + len(adj_inf_safe)
            ) == len(adj_unrevealed):
                for cell in adj_unrevealed:
                    self.remaining.remove(cell)
                    self.safe.add(cell)
        
        return len(self.safe) + len(self.revealed), triggered_mines
        

def main():
    width = 20
    height = 20
    iters = 1000
    for i in range(1, 3 * width * height // 10):
        wins = 0
        avg_triggered_mines = 0
        for _ in range(iters):
            mb = MineBoard(width, height, num_bombs=i)
            bot = SimpleLogicBot(mb)

            num_opened, triggered_mines = bot.play_board()

            avg_triggered_mines += triggered_mines

            if num_opened == width * height - i:
                wins += 1

        print(f"{i} bombs | {100*wins/iters:.2f}% WR | {avg_triggered_mines/iters:.2f} TM | {100*avg_triggered_mines/iters/i:.2f}% MTR")



if __name__ == "__main__":
    main()
