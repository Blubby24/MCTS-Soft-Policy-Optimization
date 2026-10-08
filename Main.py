from Games import *
from MCTrees import *

def play_mcts_vs_random(mcts_iters=1000, verbose=False):
    game = TicTacToe()
    state = game.initial_state()
    mcts = MCTS(game=game, iterations=mcts_iters)

    while not game.is_terminal(state):
        if state["player"] == 1:
            action = mcts.choose(state, verbose=verbose)
            print(f"MCTS (X) chooses action {action}")
        else:
            action = random.choice(game.legal_actions(state))
            print(f"Random (O) chooses action {action}")
        state = game.next_state(state, action)
        game.pretty_print(state)

    result = game.game_result(state)
    if result == 1:
        print("X (MCTS) wins!")
    elif result == -1:
        print("O (Random) wins!")
    else:
        print("Draw!")


def play_contracting_mcts_vs_random(iters=500):
    game = TicTacToe()
    mcts = ContractingMCTS(game, iterations=iters, gamma=0.97)
    state = game.initial_state()

    while not game.is_terminal(state):
        if state["player"] == 1:
            action = mcts.choose(state)
            print(f"ContractingMCTS (X) chooses {action}")
        else:
            action = random.choice(game.legal_actions(state))
            print(f"Random (O) chooses {action}")
        state = game.next_state(state, action)
        game.pretty_print(state)

    res = game.game_result(state)
    print("Result:", "X wins" if res == 1 else "O wins" if res == -1 else "Draw")


def play_game(game, player1, player2, iters=500, verbose=False):
    state = game.initial_state()
    players = {1: player1, -1: player2}

    while not game.is_terminal(state):
        current_player = state["player"]
        agent = players[current_player]

        if isinstance(agent, MCTS):
            action = agent.choose(state)
        elif agent == "random":
            action = random.choice(game.legal_actions(state))
        else:
            raise ValueError("Unknown agent type.")

        state = game.next_state(state, action)
        if verbose:
            game.pretty_print(state)

    return game.game_result(state)

def compare_all_methods_on_game(
    game_class,
    num_games=60,
    iters=300,
    gamma=0.97,
    h=3,
    m=2,
    methods_to_run=("Standard", "Contracting", "MultiStep", "FRAP", "BackupTree", "PolicyOpt", "RePReL"),
):
    game = game_class()

    # Initialize all possible agent types
    all_agents = {
        "Standard": MCTS(game, iterations=iters, exploration=math.sqrt(2.0)),
        "Contracting": ContractingMCTS(game, iterations=iters, exploration=math.sqrt(2.0), gamma=gamma),
        "MultiStep": MultiStepMCTS(game, iterations=iters, exploration=math.sqrt(2.0), gamma=gamma, h=h, m=m),
        "FRAP": FRAPMCTS(game, iterations=iters),
        "BackupTree": BackupTreeMCTS(game, iterations=iters),
        "PolicyOpt": MCTSPolicyOpt(game, iterations=iters),
        "RePReL": RePReLMCTS(game, iterations=iters),
        "PRMRL": PRMRLMCTS(game, iterations=iters),
    }

    # Filter only requested ones
    agents = {name: all_agents[name] for name in methods_to_run if name in all_agents}

    if len(agents) < 2:
        raise ValueError("Need at least two methods to compare!")

    # Generate all unique pairs for comparison
    pairs = [(a1, a2) for i, a1 in enumerate(agents) for j, a2 in enumerate(agents) if i < j]

    results = {name: 0 for name in agents}
    draws = 0

    for a1_name, a2_name in pairs:
        print(f"\n--- {a1_name} vs {a2_name} ---")
        a1 = agents[a1_name]
        a2 = agents[a2_name]

        # Each pair plays equal share of total games
        games_per_pair = max(1, num_games // len(pairs))
        for i in range(games_per_pair):
            swap = (i % 2 == 1)
            if not swap:
                res = play_game(game, a1, a2)
            else:
                res = play_game(game, a2, a1)
                res = -res

            if res == 1:
                results[a1_name] += 1
            elif res == -1:
                results[a2_name] += 1
            else:
                draws += 1

    total = sum(results.values()) + draws
    print(f"\n=== {game_class.__name__} Results ===")
    for name in agents:
        print(f"{name:12s} wins: {results[name]}")
    print(f"Draws: {draws}")
    for name in agents:
        print(f"{name:12s} win rate: {results[name]/total:.2%}")
    print(f"Draw rate: {draws/total:.2%}")

    return results



if __name__ == "__main__":
    print("==== TicTacToe ====")
    compare_all_methods_on_game(TicTacToe, num_games=5000, iters=500,
                                gamma=0.97, h=3, m=2)

    # print("\n==== ConnectFour ====")
    # compare_all_methods_on_game(ConnectFour, num_games=30, iters=300,
    #                            gamma=0.97, h=3, m=2)


