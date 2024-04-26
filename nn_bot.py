import torch
import torch.nn as nn  # neural networks
import torch.nn.functional as F  # layers, activations and more
from tqdm import tqdm
import json
from mine_board import MineBoard


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # just define for 9 x 9 for now
        self.fc1 = nn.Linear(10, 1)  # 9 x 9 x 10 -> 9 x 9
        self.conv1 = nn.Conv2d(1, 32, 5, 1)  # 1 x 9 x 9 -> 32 x 5 x 5
        self.conv2 = nn.Conv2d(32, 64, 5, 1)  # 32 x 5 x 5 -> 64 x 1 x 1
        self.fc2 = nn.Linear(64, 81)  # 128 x 1 x 1 -> 81

        def weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(weights_init)

    def forward(self, x):
        x = F.tanh(self.fc1(x.view(-1, 81, 10)))
        x = F.relu(self.conv1(x.view(-1, 1, 9, 9)))
        x = F.relu(self.conv2(x))
        x = self.fc2(x.view(-1, 64))

        return x

    def load_data(self, path):
        data = json.load(open(path, "r"))
        data = torch.tensor(data)  # 9 x 9 x 11

        N = len(data)

        outputs = torch.zeros(N, 9, 9)
        outputs = data[:, :, :, 10]
        outputs = outputs.view(N, 81)

        num_bombs = outputs.sum(dim=1)

        outputs = outputs / num_bombs.view(N, 1)

        features = torch.zeros(N, 9, 9, 10)
        features = data[:, :, :, :10]

        data_set = torch.utils.data.TensorDataset(features.float(), outputs)
        train, val, test = torch.utils.data.random_split(
            data_set, [0.8, 0.1, 0.1])

        self.train_loader = torch.utils.data.DataLoader(
            train, batch_size=16, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val, batch_size=16, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test, batch_size=16, shuffle=True
        )

    def fit(self, epochs=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()

        try:
            for epoch in range(epochs):
                total_loss = 0
                for feature, target in tqdm(self.train_loader, leave=False):
                    optimizer.zero_grad()

                    pred = self.forward(feature)
                    loss = loss_func(pred, target)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss

                print(f"Epoch {epoch} | Training Loss {
                      total_loss.item()/len(self.train_loader):.2f}")
        except KeyboardInterrupt:
            print("Stopping Training")

    def check_outputs(self):
        for input, target in self.val_loader:
            output = F.softmax(self(input), dim=1)
            break

        for i in range(9):
            for j in range(9):
                print(f"{output[0][i * 9 + j].item():.4f}", end="  ")
            print("")

        print(output[0].sum())

        print("\n")

        for i in range(9):
            for j in range(9):
                print(f"{torch.argmax(input[0][i][j]).item():.4f}", end="  ")
            print("")


class NetBot():
    def __init__(self, model, board):
        self.model = model
        self.board = board

        self.remaining = set()
        self.revealed = set()
        self.mine = set()

        self.clue_board = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).repeat((9, 9, 1)).float()

        for row in range(board.height):
            for col in range(board.width):
                self.remaining.add((row, col))

    def make_inference(self):
        self.remaining = self.remaining.union(self.mine)
        self.mine = set()

        output = self.model(self.clue_board).squeeze(-1)
        for i in range(9):
            for j in range(9):
                prob = output[i * 9 + j]
                print(prob)
                # self.remaining case should be rare
                if prob < 0.09 or (i, j) not in self.remaining:
                    continue

                self.remaining.remove((i, j))
                self.mine.add((i, j))

    def play_board(self):
        triggered_mines = 0

        while True:
            print("running")
            row, col = None, None
            if len(self.remaining) > 0:
                row, col = self.remaining.pop()
            else:
                break

            if self.board.has_mine(row, col):
                triggered_mines += 1
                self.board.remove_mine(row, col)

                neighbors = self.board.get_neighbors(row, col)
                for nrow, ncol in neighbors:
                    clue = self.clue_board[nrow][ncol]
                    self.clue_board[nrow][ncol] = F.one_hot(
                        torch.tensor(max(clue.argmax()-1, 0), 10))

            self.revealed.add((row, col))
            self.clue_board[row][col] = F.one_hot(
                torch.tensor(self.board.board[row][col]["clue"]), 10)

            self.make_inference()

        return len(self.revealed), triggered_mines


def main():
    model = Net()
    model.load_data("test_data_gen.json")
    model.fit(epochs=5)

    mb = MineBoard(9, 9, num_bombs=1)

    net_bot = NetBot(model, mb)

    print(net_bot.play_board())


if __name__ == "__main__":
    main()