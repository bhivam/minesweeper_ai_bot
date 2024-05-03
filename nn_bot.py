import sys
from pygame.event import get
import torch
import torch.utils.data
import torch.nn as nn  # neural networks
import torch.nn.functional as F  # layers, activations and more
from tqdm import tqdm
import pickle
from mine_board import MineBoard
import os
import pygame
from time import sleep
import statsmodels.stats.api as sms


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel):
        super().__init__()
        self.skip_ld = nn.Sequential(
            # kernel size 1 and padding 0 while stride > 1, doing a weighted downsampling of each "image"
            nn.Conv2d(in_filters, out_filters, kernel),
            nn.BatchNorm2d(out_filters),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, 3, 1, 1),
            nn.BatchNorm2d(out_filters),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_filters, out_filters, kernel),
            nn.BatchNorm2d(out_filters),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        res = self.skip_ld(x)
        y += res
        return F.relu(y)


class Net2(nn.Module):

    class ConvBlock(nn.Module):
        def __init__(self, in_filters, out_filters, kernel, activation=True):
            super().__init__()
            self.activation = activation
            self.skip_ld = nn.Sequential(
                # kernel size 1 and padding 0 while stride > 1, doing a weighted downsampling of each "image"
                nn.Conv2d(in_filters, out_filters, kernel),
                nn.BatchNorm2d(out_filters),
            )

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(),
            )
            self.conv2 = (
                nn.Sequential(
                    nn.Conv2d(out_filters, out_filters, kernel),
                    nn.BatchNorm2d(out_filters),
                )
                if not activation
                else nn.Conv2d(out_filters, out_filters, kernel)
            )

        def forward(self, x):
            y = self.conv1(x)
            y = self.conv2(y)
            res = self.skip_ld(x)
            y += res
            if self.activation:
                return F.relu(y)
            return y

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(10, 10)
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.conv_collapse = nn.Sequential(nn.Conv2d(256, 1, 1))
        """

        self.conv1 = Net2.ConvBlock(10, 64, 1)
        self.conv2 = Net2.ConvBlock(64, 128, 1)
        self.conv3 = Net2.ConvBlock(128, 256, 1)
        self.conv_collapse = Net2.ConvBlock(256, 1, 1, activation=False)

        def weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(weights_init)

    def forward(self, x):
        input_shape = x.shape
        N, height, width = (
            input_shape
            if len(input_shape) == 3
            else (1, input_shape[0], input_shape[1])
        )

        x = self.embedding(x)
        x = self.conv1(x.view(N, 10, height, width))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_collapse(x)
        return x.view(N, height, width)

    def load_data(self, path):
        data = pickle.load(open(path, "rb"))
        data = torch.tensor(data)  # N x H x W

        N, height, width = data.shape

        outputs = torch.zeros(N, height, width)
        outputs = data.long().clone()
        outputs = (outputs == 10).long()

        features = data[:, :, :].long()
        features[torch.logical_or(features.long() == 10, features.long() == 11)] = 9

        data_set = torch.utils.data.TensorDataset(
            features.to(self.device), outputs.to(self.device)
        )
        train, val, test = torch.utils.data.random_split(data_set, [0.9, 0.05, 0.05])

        self.train_loader = torch.utils.data.DataLoader(
            train, batch_size=1024, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val, batch_size=1024, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test, batch_size=1024, shuffle=True
        )

    def fit(self, epochs=10):
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.to(self.device)

        try:
            for epoch in range(epochs):
                total_loss = torch.tensor(0).float()
                for feature, target in tqdm(self.train_loader, leave=False):
                    optimizer.zero_grad()

                    pred = self.forward(feature)
                    loss = loss_func(pred.flatten(), target.flatten().float())

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.cpu()

                    del loss
                    del pred

                val_loss = torch.tensor(0).float()
                for feature, target in self.val_loader:
                    with torch.no_grad():
                        pred = self.forward(feature)
                        loss = loss_func(pred.flatten(), target.flatten().float())

                        val_loss += loss.cpu()

                        del loss
                        del pred

                print(
                    f"Epoch {epoch} | Training Loss {total_loss.item()/len(self.train_loader)} | Validation Loss {val_loss.item()/len(self.val_loader)}"
                )
        except KeyboardInterrupt:
            print("Stopping Training")

        self.cpu()

    def test_model(self):
        loss_func = nn.BCEWithLogitsLoss()
        test_loss = torch.tensor(0).float()
        self.to(self.device)
        for feature, target in self.test_loader:
            with torch.no_grad():
                feature = feature.to(self.device)
                target = target.cuda(self.device)

                pred = self.forward(feature)
                loss = loss_func(pred.flatten(), target.flatten().float())

                test_loss += loss.cpu()

        return test_loss / len(self.test_loader)

    def check_outputs(self):
        x, pred, y = [], [], []

        for input, output in self.val_loader:
            x = input
            y = output

            self.to(self.device)
            input = input.to(self.device)

            pred = F.sigmoid(self(input))
            break

        h, w = x.shape[1:]

        for k in range(50, 51):
            for i in range(h):
                for j in range(w):
                    print(f"{x[k][i][j].item():.4f}", end="  ")
                print("")
            print("\n\n")

            for i in range(h):
                for j in range(w):
                    print(f"{pred[k][i][j].item():.4f}", end="  ")
                print("")
            print("\n\n")

            for i in range(h):
                for j in range(w):
                    print(f"{y[k][i][j].item():.4f}", end="  ")
                print("")
            print("\n\n")


class Net1(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def create_layer(filters, num_layers, kernel):
            layers = []
            layers.append(ConvBlock(self.in_filters, filters, kernel))
            self.in_filters = filters
            for _ in range(1, num_layers):
                layers.append(ConvBlock(self.in_filters, filters, 1))

            return nn.Sequential(*layers)

        self.in_filters = 128

        self.embedding = nn.Sequential(nn.Linear(2, 22), nn.Tanh())
        self.conv1 = nn.Sequential(
            nn.Conv2d(22, 128, 3, 1, 0), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv2 = nn.Sequential(create_layer(256, 2, 3))
        self.fc2 = nn.Linear(256, 1)

        def weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)

        self.apply(weights_init)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x.view(-1, 22, 5, 5))
        x = self.conv2(x)
        x = self.fc2(x.view(-1, 256))

        return x

    def load_data(self, path):
        data = pickle.load(open(path, "rb"))
        data = torch.tensor(data)  # N x 5 x 5

        N = len(data)

        outputs = torch.zeros(N)
        outputs = data[:, 2, 2, 0].long()
        outputs = outputs.eq(11).long()

        features = torch.zeros(N, 5, 5, 2)
        features = data[:, :, :]

        features[features.long() == 11] = 9

        data_set = torch.utils.data.TensorDataset(
            features.to(self.device), outputs.to(self.device)
        )
        train, val, test = torch.utils.data.random_split(data_set, [0.9, 0.05, 0.05])

        self.train_loader = torch.utils.data.DataLoader(
            train, batch_size=1024, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val, batch_size=1024, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test, batch_size=1024, shuffle=True
        )

    def fit(self, epochs=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        loss_func = nn.BCEWithLogitsLoss()
        self.to(self.device)

        try:
            for epoch in range(epochs):
                total_loss = torch.tensor(0).float()
                for feature, target in tqdm(self.train_loader, leave=False):
                    optimizer.zero_grad()

                    pred = self.forward(feature)
                    loss = loss_func(pred.flatten(), target.flatten().float())

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.cpu()

                    del loss
                    del pred

                val_loss = torch.tensor(0).float()
                for feature, target in self.val_loader:
                    with torch.no_grad():
                        pred = self.forward(feature)
                        loss = loss_func(pred.flatten(), target.flatten().float())

                        val_loss += loss.cpu()

                        del loss
                        del pred

                print(
                    f"Epoch {epoch} | Training Loss {total_loss.item()/len(self.train_loader)} | Validation Loss {val_loss.item()/len(self.val_loader)}"
                )
        except KeyboardInterrupt:
            print("Stopping Training")

        self.cpu()

    def test_model(self):
        loss_func = nn.BCEWithLogitsLoss()
        test_loss = torch.tensor(0).float()
        self.to(self.device)
        for feature, target in self.test_loader:
            with torch.no_grad():
                feature = feature.to(self.device)
                target = target.cuda(self.device)

                pred = self.forward(feature)
                loss = loss_func(pred.flatten(), target.flatten().float())

                test_loss += loss.cpu()

        return test_loss / len(self.test_loader)

    def check_outputs(self):
        x, pred, y = [], [], []

        for input, output in self.val_loader:
            x = input
            y = output

            self.to(self.device)
            input = input.to(self.device)

            pred = F.sigmoid(self(input))
            break

        for k in range(50, 60):
            print(f"Predicted Bomb Chance: {100*pred[k].item():.2f}%")
            print(f"Bomb Exists: {y[k].bool()}")
            print()

            for i in range(5):
                for j in range(5):
                    print(f"{x[k][i][j][0].item():.4f}", end="  ")
                print("")

            print("")
            for i in range(5):
                for j in range(5):
                    print(f"{F.sigmoid(x[k][i][j][1]).item():.4f}", end="  ")
                print("")

            print("\n\n")


class NetBot:
    def __init__(self, model, board: MineBoard, model_type, debug=False):
        self.model = model
        self.board = board
        self.debug = debug
        self.model_type = model_type

        assert model_type == "net1" or model_type == "net2"

        self.calculated = dict()
        self.remaining = set()
        self.revealed = set()
        self.next_move = None
        self.unrevealed_edge = set()

        self.correct_predictions = 0

        self.clue_board = torch.ones((board.height, board.width)) * 9

        for row in range(board.height):
            for col in range(board.width):
                self.remaining.add((row, col))

    def get_cell_state(self, cell):
        row, col = cell
        if cell in self.revealed:
            return int(self.clue_board[row][col])
        elif row < 0 or row >= self.board.height or col < 0 or col >= self.board.width:
            return 10  # off board

        elif self.board.has_mine(*cell):
            return 11  # has mine
        else:
            return 9  # unrevealed

    def get_example(self, cell, radius=2, probs=None):
        if probs is None:
            probs = self.calculated
        return [
            [
                [
                    float(self.get_cell_state((i, j))),
                    probs[(i, j)].item() if (i, j) in probs and (i, j) != cell else -1,
                ]
                for i in range(cell[0] - radius, cell[0] + radius + 1)
            ]
            for j in range(cell[1] - radius, cell[1] + radius + 1)
        ]

    def get_examples(self, last_move, radius=2):
        examples = []

        # get last move, get all adjacent unrevealed tiles
        for ncell in self.board.get_neighbors(*last_move):
            if ncell not in self.revealed:  # if neighbor not revealed
                examples.append(self.get_example(ncell, radius=radius))

        return examples

    def get_cell_state_net2(self, cell, edge=True):
        if cell in self.revealed:
            return int(self.clue_board[cell[0]][cell[1]])
        elif self.board.has_mine(*cell) and edge:
            return 10  # has mine
        elif not self.board.has_mine(*cell) and edge:
            return 11  # has mine
        else:
            return 9  # unrevealed

    def get_example_net2(self, for_input=False):
        example = [
            [
                self.get_cell_state_net2((i, j), edge=False)
                for j in range(self.board.width)
            ]
            for i in range(self.board.height)
        ]

        if for_input:
            return example

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
        last_move = self.next_move  # this is none when we have no calculated cells

        # remove stale probabilities from calculated cache
        prev_calculated = self.calculated.copy()

        if last_move is not None:
            if self.model_type == "net1":
                del self.calculated[last_move]
                last_move_impacted = self.board.get_neighbors(
                    last_move[0], last_move[1], 2
                )
                for row, col in last_move_impacted:
                    if (row, col) in self.calculated:
                        del self.calculated[row, col]
            else:
                self.calculated = dict()

            self.unrevealed_edge.remove(last_move)
            for row, col in self.board.get_neighbors(*last_move):
                if (row, col) not in self.revealed:
                    self.unrevealed_edge.add((row, col))
        else:
            for row, col in self.revealed:
                for nrow, ncol in self.board.get_neighbors(row, col):
                    if (nrow, ncol) not in self.revealed:
                        self.unrevealed_edge.add((nrow, ncol))

        if self.model_type == "net1":
            for row, col in self.unrevealed_edge:
                if (row, col) not in self.calculated:
                    self.calculated[(row, col)] = self.model(
                        torch.tensor(
                            self.get_example((row, col), probs=prev_calculated)
                        )
                        .to(self.model.device)
                        .float()
                    ).cpu()
        else:  # "net2"
            input = self.get_example_net2(for_input=True)
            output = self.model(torch.tensor(input).to(self.model.device)).squeeze(0)
            for row, col in self.unrevealed_edge:
                self.calculated[(row, col)] = output[row][col].float()
            if self.debug:
                for i in range(self.board.height):
                    for j in range(self.board.width):
                        print(
                            f"{self.calculated[(i, j)] if (i, j) in self.calculated else 0:.2f}",
                            end=" ",
                        )
                    print()
                print("\n\n")

                for i in range(self.board.height):
                    for j in range(self.board.width):
                        print(f"{input[i][j]:.2f}", end=" ")
                    print()
                print("\n\n")

                for i in range(self.board.height):
                    for j in range(self.board.width):
                        if (i, j) in self.unrevealed_edge:
                            print(
                                f"{1 if self.board.has_mine(i, j) else -1:.2f}", end=" "
                            )
                        else:
                            print(f"{0:.2f}", end=" ")
                    print()

        self.next_move = sorted(self.calculated.items(), key=lambda x: x[1])[0][0]
        self.correct_predictions += 0 if self.board.has_mine(*self.next_move) else 1
        self.remaining.remove(self.next_move)

        # Display probabilities for neighbors along with clue information
        if self.debug == True:
            screen.fill((0, 0, 0))
            block_dim = min(800 // self.board.height, 800 // self.board.width)
            font = pygame.font.SysFont("arial", block_dim // 3)
            for row in range(self.board.height):
                for col in range(self.board.width):
                    tile = pygame.Rect(
                        col * block_dim,
                        row * block_dim,
                        block_dim,
                        block_dim,
                    )
                    text = None
                    if (row, col) == self.next_move:
                        ind_color = (
                            (207, 51, 27)
                            if self.board.has_mine(row, col)
                            else (26, 194, 23)
                        )
                        pygame.draw.rect(screen, ind_color, tile, 0)
                        text = f"{int(100*F.sigmoid(self.calculated[(row, col)]))}%"

                    elif (row, col) in self.calculated:
                        pygame.draw.rect(screen, (180, 180, 180), tile, 0)
                        text = f"{int(100*F.sigmoid(self.calculated[(row, col)]))}%"
                    elif int(self.clue_board[row][col]) == 9:
                        pygame.draw.rect(screen, (100, 100, 100), tile, 0)
                    elif int(self.clue_board[row][col]) == 0:
                        pygame.draw.rect(screen, (230, 230, 230), tile, 0)
                    else:
                        clue_colors = [
                            (224, 243, 176),
                            (254, 235, 201),
                            (255, 255, 176),
                            (179, 226, 221),
                            (204, 236, 239),
                            (191, 213, 232),
                            (221, 212, 232),
                            (253, 222, 238),
                        ]
                        pygame.draw.rect(
                            screen,
                            clue_colors[int(self.clue_board[row][col]) - 1],
                            tile,
                            0,
                        )
                        text = f"{int(self.clue_board[row][col])}"
                    pygame.draw.rect(screen, (0, 0, 0), tile, 1)

                    if text is not None:
                        text = font.render(text, True, (0, 0, 0))
                        tile.move_ip((block_dim // 3, block_dim // 3))
                        screen.blit(text, tile)

    def play_board(self, collect_data=False, mistakes_only=False, format="net1"):
        assert collect_data != False or mistakes_only != True
        self.revealed_before_boom = 0
        triggered_mines = 0
        num_bombs = self.board.get_num_mines()

        examples = []
        paused = False

        if self.debug == True:
            global screen
            pygame.init()
            screen = pygame.display.set_mode((800, 800))
            screen.fill((200, 200, 200))

        while True:
            if self.debug:
                sleep(0.5)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                    elif event.type == pygame.KEYDOWN:
                        paused = not paused

                pygame.display.update()

            if paused:
                continue

            if self.debug:
                print(
                    f"revealed {len(self.revealed)} | remaining {len(self.remaining)} | total {len(self.revealed) + len(self.remaining) + 1 if self.next_move is not None else 0}"
                )

            if collect_data and self.next_move is not None:
                if not mistakes_only and format == "net1":
                    examples.extend(self.get_examples(self.next_move))
                elif not mistakes_only and format == "net2":
                    examples.append(self.get_example_net2())
                elif self.board.has_mine(*self.next_move):
                    examples.append(self.get_example(self.next_move))

            row, col = None, None
            if len(self.remaining) == num_bombs:
                break
            elif self.next_move is not None:
                row, col = self.next_move
            elif len(self.remaining) > 0:
                row, col = self.board.start_cell
                self.remaining.remove((row, col))

            if self.board.has_mine(row, col):
                if triggered_mines == 0:
                    self.revealed_before_boom = len(self.revealed)
                triggered_mines += 1
                self.board.remove_mine(row, col)
                num_bombs -= 1

                neighbors = self.board.get_neighbors(row, col)
                for nrow, ncol in neighbors:
                    self.clue_board[nrow][ncol] = self.board.board[nrow][ncol]["clue"]

            self.revealed.add((row, col))
            self.clue_board[row][col] = self.board.board[row][col]["clue"]

            if num_bombs == len(self.remaining):
                break

            self.make_inference()

        if self.revealed_before_boom == 0:
            self.revealed_before_boom = len(self.revealed)

        return (
            self.revealed_before_boom,
            triggered_mines,
            self.correct_predictions,
            examples,
        )


def collect_data(model, height, width, bombs, iters, path, model_type, format):
    examples = []
    for _ in tqdm(range(iters)):
        mb = MineBoard(height, width, num_bombs=bombs)
        net_bot = NetBot(model, mb, model_type)
        _, _, _, examples_i = net_bot.play_board(collect_data=True, format=format)
        examples.extend(examples_i)

    pickle.dump(examples, open(path, "wb"))


def collect_mixed_data(model, iters, path, model_type, format, get_mistakes=False):
    examples = []
    for i in tqdm(range(iters)):
        p = (i % 25 + 6) / 100
        mb = MineBoard(9, 9, p=p)
        net_bot = NetBot(model, mb, model_type)
        _, _, _, examples_i = net_bot.play_board(
            collect_data=True, mistakes_only=get_mistakes, format=format
        )
        examples.extend(examples_i)

    pickle.dump(examples, open(path, "wb"))


def test_bot(
    model, model_type, height=9, width=9, min_bombs=1, max_bombs=10, iters=100
):
    for bombs in range(min_bombs, max_bombs + 1):
        wins = []
        mt = []
        correct_rates = []
        revealed = []
        for _ in tqdm(range(iters), leave=False):
            mb = MineBoard(height, width, num_bombs=bombs)
            net_bot = NetBot(model, mb, model_type)
            revealed_i, mt_i, correct_predictions, _ = net_bot.play_board()

            if mt_i == 0:
                wins.append(1)
            else:
                wins.append(0)
            mt.append(mt_i)
            revealed.append(revealed_i)
            correct_rates.append(1 - mt_i / (height * width + mt_i - bombs))

        wins_conf = sms.DescrStatsW(wins).tconfint_mean()
        revealed_conf = sms.DescrStatsW(revealed).tconfint_mean()
        cr_conf = sms.DescrStatsW(correct_rates).tconfint_mean()
        mt_conf = sms.DescrStatsW(mt).tconfint_mean()

        format_ci = (
            lambda ci, is_percent: f"{(100 if is_percent else 1)*(ci[0]+ci[1])/2:.2f}±{(100 if is_percent else 1)*abs(ci[0]-(ci[0]+ci[1])/2):.2f}"
        )

        print(
            f"Bombs: {bombs} |Average MT: {format_ci(mt_conf, False)}| Average Revealed: {format_ci(revealed_conf, False)} | WR: {format_ci(wins_conf, True)}%| Average CPR: {format_ci(cr_conf, True)}%"
        )


def play_with_visual(model, height, width, bombs, model_type):
    mb = MineBoard(height, width, num_bombs=bombs)
    net_bot = NetBot(model, mb, model_type, debug=True)
    net_bot.play_board()


def main():
    argv = sys.argv
    argc = len(argv)

    assert argc > 1

    if argv[1] == "train":
        # you can keyboard interrupt during training, your model will be saved.
        # data_source model_output_path epochs
        if argc != 6:
            print(
                "Usage: python3 nn_bot.py train <data path> <model path> <epochs> <model type>"
            )
            sys.exit()

        _, _, data_path, model_path, epochs, model_type = argv

        assert model_type == "net1" or model_type == "net2"

        if not os.path.exists(data_path):
            print("data path does not exist")
            sys.exit(1)

        epochs = int(epochs)

        model = Net1() if model_type == "net1" else Net2()
        model.load_data(data_path)
        model.fit(epochs=epochs)
        model.check_outputs()
        torch.save(model, model_path)
        print(f"Final Test Loss: {model.test_model()}")

    elif argv[1] == "finetune":
        if argc != 6:
            print(
                "Usage: python3 nn_bot.py train <data path> <old model path> <new model path> <epochs>"
            )

        _, _, data_path, old_model_path, new_model_path, epochs = argv

        if not os.path.exists(data_path):
            print("data path does not exist")
            sys.exit(1)

        if not os.path.exists(old_model_path):
            print("old model path does not exist")
            sys.exit(1)

        epochs = int(epochs)

        model = torch.load(old_model_path)
        model.load_data(data_path)
        model.fit(epochs=epochs)
        torch.save(model, new_model_path)
        print(f"Final Test Loss: {model.test_model()}")

    elif argv[1] == "visual":
        # which model, height, width, bombs
        if argc != 7:
            print(
                "Usage: python3 nn_bot.py visual <model_path> <board height> <board width> <# of bombs> <model_type>"
            )
            sys.exit(1)
        _, _, model_path, height, width, num_bombs, model_type = argv

        if not os.path.exists(model_path):
            print("path to model spec does not exist")
            sys.exit(1)

        model, height, width, num_bombs = (
            torch.load(model_path),
            int(height),
            int(width),
            int(num_bombs),
        )
        model.to(model.device)
        model.eval()

        print(f"Model: {model_path} | HxW: {height}x{width} | # of Mines: {num_bombs}")
        play_with_visual(model, height, width, num_bombs, model_type)

    elif argv[1] == "benchmark":
        if argc != 9:
            print(
                "Usage: python3 nn_bot.py visual <model_path> <board height> <board width> <min # of bombs> <max # of bombs> <iters> <model type>"
            )
            sys.exit(1)
        _, _, model_path, height, width, min_bombs, max_bombs, iters, model_type = argv

        if not os.path.exists(model_path):
            print("path to model spec does not exist")
            sys.exit(1)

        model, height, width, min_bombs, max_bombs, iters = (
            torch.load(model_path),
            int(height),
            int(width),
            int(min_bombs),
            int(max_bombs),
            int(iters),
        )
        model.to(model.device)
        model.eval()

        print(
            f"Model: {model_path} | HxW: {height}x{width} | Bomb Range: {min_bombs}-{max_bombs} | {iters} Iterations"
        )
        test_bot(model, model_type, height, width, min_bombs, max_bombs, iters)
    elif argv[1] == "old_datagen":
        if argc != 10:
            print(
                "Usage: python3 nn_bot.py old_datagen <model path> <board height> <board width> <# of bombs> <iters> <data path> <model_type> <format>"
            )
            sys.exit(1)

        (
            _,
            _,
            model_path,
            height,
            width,
            num_bombs,
            iters,
            data_path,
            model_type,
            format,
        ) = argv

        height, width, num_bombs, iters = (
            int(height),
            int(width),
            int(num_bombs),
            int(iters),
        )

        if not os.path.exists(model_path):
            print("path to model spec does not exist")
            sys.exit(1)

        model = torch.load(model_path)
        model.to(model.device)
        model.eval()
        collect_data(
            model, height, width, num_bombs, iters, data_path, model_type, format
        )

    elif argv[1] == "datagen":
        if argc != 7:
            print(
                "Usage: python3 nn_bot.py datagen <model path> <iters> <data path> <model type> <format>"
            )
            sys.exit(1)

        _, _, model_path, iters, data_path, model_type, format = argv
        iters = int(iters)

        if not os.path.exists(model_path):
            print("path to model spec does not exist")
            sys.exit(1)

        model = torch.load(model_path)
        model.to(model.device)
        model.eval()
        collect_mixed_data(model, iters, data_path, model_type, format)
    else:
        print(
            "Usage: python3 nn_bot.py [fintune, train, visual, benchmark, datagen, old_datagen]"
        )


if __name__ == "__main__":
    main()
