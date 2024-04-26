import torch
import torch.utils.data
import torch.nn as nn  # neural networks
import torch.nn.functional as F  # layers, activations and more
from tqdm import tqdm
import json
from mine_board import MineBoard
import os


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # just define for 9 x 9 for now
        self.fc1 = nn.Linear(10, 1)  # 9 x 9 x 10 -> 9 x 9
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)  # 1 x 9 x 9 -> 32 x 9 x 9
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # 32 x 9 x 9 -> 64 x 9 x 9
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)  # 64 x 9 x 9 -> 128 x 9 x 9
        self.conv4 = nn.Conv2d(128, 256, 5, 1)  # 128 x 9 x 9 -> 256 x 5 x 5
        self.conv5 = nn.Conv2d(256, 512, 5, 1)  # 256 x 5 x 5 -> 512 x 1 x 1
        self.fc2 = nn.Linear(512, 81)  # 128 x 1 x 1 -> 81

        def weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(weights_init)

    def forward(self, x):
        x = F.tanh(self.fc1(x.view(-1, 81, 10)))
        x = F.relu(self.conv1(x.view(-1, 1, 9, 9)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.fc2(x.view(-1, 512))

        return x

    def load_data(self, path):
        data = json.load(open(path, "r"))
        data = torch.tensor(data)  # N x 9 x 9 x 11

        N = len(data)

        outputs = torch.zeros(N, 9, 9)
        outputs = data[:, :, :, 10]
        outputs = outputs.view(N, 81)
        # flip ones and zeros to get safe spots
        outputs = -outputs + 1

        features = torch.zeros(N, 9, 9, 10)
        features = data[:, :, :, :10]

        data_set = torch.utils.data.TensorDataset(features.float(), outputs)
        train, val, test = torch.utils.data.random_split(data_set, [0.8, 0.1, 0.1])

        self.train_loader = torch.utils.data.DataLoader(
            train, batch_size=256, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(val, batch_size=16, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            test, batch_size=256, shuffle=True
        )

    def fit(self, epochs=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_func = nn.BCEWithLogitsLoss()
        self.to(self.device)

        try:
            for epoch in range(epochs):
                total_loss = torch.tensor(0).float()
                for feature, target in tqdm(self.train_loader, leave=False):
                    feature = feature.to(self.device)
                    target = target.cuda(self.device)

                    optimizer.zero_grad()

                    pred = self.forward(feature)
                    loss = loss_func(pred.flatten(), target.flatten().float())

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.cpu()

                val_loss = torch.tensor(0).float()
                for feature, target in self.val_loader:
                    feature = feature.to(self.device)
                    target = target.cuda(self.device)

                    optimizer.zero_grad()

                    pred = self.forward(feature)
                    loss = loss_func(pred.flatten(), target.flatten().float())

                    val_loss += loss.cpu()
                print(
                    f"Epoch {epoch} | Training Loss {total_loss.item()/len(self.train_loader)} | Validation Loss {val_loss.item()/len(self.train_loader)}"
                )
        except KeyboardInterrupt:
            print("Stopping Training")

        self.cpu()

    def check_outputs(self):
        x, pred, y = [], [], []

        for input, output in self.val_loader:
            x = input
            y = output
            pred = F.sigmoid(self(input))
            break

        for i in range(9):
            for j in range(9):
                print(f"{pred[0][i * 9 + j].item():.4f}", end="  ")
            print("")

        print("\n")
        for i in range(9):
            for j in range(9):
                print(f"{y[0][i * 9 + j].item():.4f}", end="  ")
            print("")

        print("\n")

        for i in range(9):
            for j in range(9):
                print(f"{torch.argmax(x[0][i][j]).item():.4f}", end="  ")
            print("")


class NetBot:
    def __init__(self, model, board: MineBoard):
        self.model = model
        self.board = board

        self.remaining = set()
        self.revealed = set()
        self.mine = set()
        self.safe = set()

        self.clue_board = (
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).repeat((9, 9, 1)).float()
        )

        for row in range(board.height):
            for col in range(board.width):
                self.remaining.add((row, col))

    def make_inference(self):
        output = F.sigmoid(self.model(self.clue_board).squeeze(0)).round()
        print(output)
        self.remaining = self.remaining.union(self.mine).union(self.safe)
        self.mine = set()
        self.safe = set()

        unopened_neighbors = set()
        # We want to look at all cells that are unopened neighbors of opened cells
        for row, col in self.revealed:
            for nrow, ncol in self.board.get_neighbors(row, col):
                if (nrow, ncol) not in self.revealed:
                    unopened_neighbors.add((nrow, ncol))
        # Let's classify each of these as being safe or being a bomb
        for row, col in unopened_neighbors:
            prob = int(output[row * 9 + col])
            if prob == 1:
                self.remaining.remove((row, col))
                self.safe.add((row, col))
            else:
                self.remaining.remove((row, col))
                self.mine.add((row, col))

    def play_board(self):
        triggered_mines = 0

        while True:
            print(
                f"revealed {len(self.revealed)} | remaining {len(self.remaining)} | mines {len(self.mine)} | safe {len(self.safe)} | total {len(self.revealed) + len(self.remaining) + len(self.mine) + len(self.safe)}"
            )

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
                for nrow, ncol in neighbors:
                    clue = self.clue_board[nrow][ncol]
                    self.clue_board[nrow][ncol] = F.one_hot(
                        torch.tensor(max(int(clue.argmax() - 1), 0)), 10
                    )

            self.revealed.add((row, col))
            self.clue_board[row][col] = F.one_hot(
                torch.tensor(self.board.board[row][col]["clue"]), 10
            )

            self.make_inference()

        return len(self.revealed), triggered_mines


def main():
    model = None
    if not os.path.exists("model.pt"):
        model = Net()
        model.load_data("test_data_gen.json")
        model.fit(epochs=200)
        model.check_outputs()
        torch.save(model, "model.pt")
    else:
        model = torch.load("model.pt")

    mb = MineBoard(9, 9, num_bombs=10)

    net_bot = NetBot(model, mb)

    print(net_bot.play_board())


if __name__ == "__main__":
    main()
