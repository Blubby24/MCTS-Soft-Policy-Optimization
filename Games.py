from copy import deepcopy
from typing import List, Any, Optional


class ConnectFour:
    ROWS, COLS = 6, 7

    def initial_state(self):
        return {"board": [[0] * self.COLS for _ in range(self.ROWS)], "player": 1}

    def current_player(self, state):
        return state["player"]

    def legal_actions(self, state):
        return [c for c in range(self.COLS) if state["board"][0][c] == 0]

    def next_state(self, state, action):
        new = deepcopy(state)
        for r in reversed(range(self.ROWS)):
            if new["board"][r][action] == 0:
                new["board"][r][action] = new["player"]
                new["player"] = -new["player"]
                return new
        raise ValueError("Column full")

    def is_terminal(self, state):
        return self._winner(state) is not None or all(state["board"][0][c] != 0 for c in range(self.COLS))

    def game_result(self, state):
        winner = self._winner(state)
        if winner is not None:
            return winner
        return 0  # draw

    def _winner(self, state):
        b = state["board"]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                player = b[r][c]
                if player == 0:
                    continue
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if all(
                            0 <= r + i * dr < self.ROWS and 0 <= c + i * dc < self.COLS and b[r + i * dr][
                                c + i * dc] == player
                            for i in range(4)
                    ):
                        return player
        return None

    def pretty_print(self, state):
        symbols = {1: "X", -1: "O", 0: "."}
        for row in state["board"]:
            print(" ".join(symbols[c] for c in row))
        print("1 2 3 4 5 6 7")
        print(f"Next: {'X' if state['player'] == 1 else 'O'}\n")


class TicTacToe:
    def __init__(self):
        pass

    @staticmethod
    def initial_state():
        return {
            "board": [0] * 9,
            "player": 1
        }

    @staticmethod
    def current_player(state) -> int:
        return state["player"]

    @staticmethod
    def legal_actions(state) -> List[int]:
        return [i for i, v in enumerate(state["board"]) if v == 0]

    @staticmethod
    def next_state(state, action) -> dict:
        new = deepcopy(state)
        if new["board"][action] != 0:
            raise ValueError("Illegal action")
        new["board"][action] = new["player"]
        new["player"] = -new["player"]
        return new

    @staticmethod
    def is_terminal(state) -> bool:
        board = state["board"]
        wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                (0, 4, 8), (2, 4, 6)]
        for (a, b, c) in wins:
            s = board[a] + board[b] + board[c]
            if abs(s) == 3:
                return True
        if all(v != 0 for v in board):
            return True
        return False

    @staticmethod
    def game_result(state) -> int:
        board = state["board"]
        wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                (0, 4, 8), (2, 4, 6)]
        for (a, b, c) in wins:
            s = board[a] + board[b] + board[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        # draw
        return 0

    @staticmethod
    def pretty_print(state):
        sym = {1: "X", -1: "O", 0: " "}
        b = state["board"]
        rows = []
        for r in range(3):
            rows.append("|".join(sym[b[r * 3 + c]] for c in range(3)))
        print("\n-----\n".join(rows))
        print(f"Next player: {'X (1)' if state['player'] == 1 else 'O (-1)'}\n")

