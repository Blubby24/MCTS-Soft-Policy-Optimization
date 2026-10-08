import math
import random
from copy import deepcopy
from typing import List, Any, Optional


class MCTSNode:
    def __init__(self, state: Any, parent: Optional["MCTSNode"], player_just_moved: int):
        self.state = state
        self.parent = parent
        self.player_just_moved = player_just_moved
        self.children: List[MCTSNode] = []
        self.untried_actions: List[Any] = []
        self.visits = 0
        self.wins = 0.0

    def uct_select_child(self, exploration: float = math.sqrt(2.0)) -> "MCTSNode":
        log_parent_visits = math.log(self.visits) if self.visits > 0 else 0.0

        def uct_value(child: MCTSNode):
            if child.visits == 0:
                return float("inf")
            win_rate = child.wins / child.visits
            return win_rate + exploration * math.sqrt(log_parent_visits / child.visits)

        return max(self.children, key=uct_value)


class MCTS:
    def __init__(self, game, iterations: int = 1000, exploration: float = math.sqrt(2.0)):
        """
        game: object implementing the interface described above
        iterations: number of MCTS iterations per move
        exploration: exploration constant in UCT
        """
        self.game = game
        self.iterations = iterations
        self.exploration = exploration

    def choose(self, root_state, verbose=False) -> Any:
        root_node = MCTSNode(state=deepcopy(root_state), parent=None,
                             player_just_moved=-self.game.current_player(root_state))
        root_node.untried_actions = self.game.legal_actions(root_state)

        for i in range(self.iterations):
            node = root_node
            state = deepcopy(root_state)

            while node.untried_actions == [] and node.children:
                node = node.uct_select_child(self.exploration)
                state = self.game.next_state(state, self._action_to_child(node, state))

            if node.untried_actions:
                action = random.choice(node.untried_actions)
                state = self.game.next_state(state, action)
                child_node = MCTSNode(state=deepcopy(state), parent=node,
                                      player_just_moved=self.game.current_player(state) * -1)
                child_node.untried_actions = self.game.legal_actions(state)
                node.untried_actions.remove(action)
                node.children.append(child_node)
                node = child_node

            rollout_state = deepcopy(state)
            result = self._simulate(rollout_state)

            self._backpropagate(node, result)

        best_child = max(root_node.children, key=lambda c: c.visits)
        best_action = self._action_to_child(best_child, root_state)
        if verbose:
            print(
                f"[MCTS] Iterations: {self.iterations}. Best action visits: {best_child.visits}, win_score: {best_child.wins}")
        return best_action

    def _action_to_child(self, child_node: MCTSNode, parent_state) -> Any:
        for a in self.game.legal_actions(parent_state):
            if self.game.next_state(deepcopy(parent_state), a) == child_node.state:
                return a
        raise RuntimeError("Could not map child node to an action from parent state")

    def _simulate(self, state) -> int:
        while not self.game.is_terminal(state):
            actions = self.game.legal_actions(state)
            action = random.choice(actions)
            state = self.game.next_state(state, action)
        return self.game.game_result(state)

    def _backpropagate(self, node: MCTSNode, result: int):
        while node is not None:
            node.visits += 1
            if result == 0:
                node.wins += 0.5
            elif result == node.player_just_moved:
                node.wins += 1.0
            node = node.parent


class ContractingMCTS(MCTS):
    """
    MCTS variant that performs γʰ-contracting backups.
    Each node receives discounted credit from deeper levels
    (reuse of tree byproducts as in hm-PI from Efroni et al., 2019).
    """

    def __init__(self, game, iterations=1000, exploration=math.sqrt(2.0), gamma=0.97):
        super().__init__(game, iterations, exploration)
        self.gamma = gamma

    def _backpropagate(self, node, result):
        depth_result = 0
        while node is not None:
            node.visits += 1
            reward = 0.5 if result == 0 else (1.0 if result == node.player_just_moved else 0.0)
            node.wins += (self.gamma ** depth_result) * reward
            depth_result += 1
            node = node.parent


class MultiStepMCTS(MCTS):
    """
    Multi-step lookahead MCTS:
      - h: lookahead depth (number of plies to follow/lookahead before bootstrapping)
      - m: number of extra random playout steps for the m-return after the lookahead horizon
      - gamma: discount factor for combining rewards across the h horizon and m-return
    Implementation details / approximations:
      - During simulation, we first follow up to h plies (randomly selecting actions that may be chosen
        from the expanded subtree when available). We accumulate discounted rewards during these h plies.
      - If we hit terminal before h, we use the exact terminal payoff.
      - If not terminal after h plies, we continue for m random playout steps and use that as an m-step
        return (partial evaluation). The final returned value is sum_{t=0..h-1} gamma^t r_t + gamma^h * m_return.
      - Backpropagation adds this scalar return (from player-1-perspective) to value, storing sign relative
        to node.player_just_moved (so higher is better for that node's player).
    """

    def __init__(self, game, iterations: int = 1000, exploration: float = math.sqrt(2.0),
                 gamma: float = 0.97, h: int = 3, m: int = 2):
        super().__init__(game, iterations, exploration)
        self.h = max(1, h)
        self.m = max(0, m)
        self.gamma = gamma

    def _rollout_with_lookahead(self, state) -> float:
        """
        Performs up-to-h lookahead steps (random) accumulating discounted rewards, then
        bootstraps with an m-step random return. Returns scalar reward from player-1 perspective.
        """
        cur = deepcopy(state)
        total = 0.0
        discount = 1.0

        for step in range(self.h):
            if self.game.is_terminal(cur):
                return self.game.game_result(cur) * discount + total  # immediate terminal result
            a = random.choice(self.game.legal_actions(cur))

            cur = self.game.next_state(cur, a)
            discount *= self.gamma

        # after h steps, if terminal:
        if self.game.is_terminal(cur):
            return self.game.game_result(cur) * discount + total

        # otherwise do m-step random rollout for partial evaluation
        bootstrap = 0.0
        cur2 = deepcopy(cur)
        discount2 = 1.0
        for j in range(self.m):
            if self.game.is_terminal(cur2):
                bootstrap = self.game.game_result(cur2) * discount2
                break
            a2 = random.choice(self.game.legal_actions(cur2))
            cur2 = self.game.next_state(cur2, a2)
            # no step rewards assumed
            discount2 *= self.gamma

        total += (self.gamma ** self.h) * (bootstrap if bootstrap != 0 else 0.0)
        return total

    def choose(self, root_state):
        root = MCTSNode(deepcopy(root_state), None, -self.game.current_player(root_state))
        root.untried_actions = self.game.legal_actions(root_state)

        for _ in range(self.iterations):
            node = root
            state = deepcopy(root_state)

            # Selection
            while not node.untried_actions and node.children:
                node = node.uct_select_child(self.exploration)
                action = self._action_to_child(node, state)
                state = self.game.next_state(state, action)

            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                state = self.game.next_state(state, action)
                child = MCTSNode(deepcopy(state), node, -self.game.current_player(state))
                child.untried_actions = self.game.legal_actions(state)
                node.children.append(child)
                node = child

            value = self._rollout_with_lookahead(state)
            self._backpropagate(node, value)

        # select best by visits
        best = max(root.children, key=lambda c: c.visits)
        return self._action_to_child(best, root_state)


class FRAPMCTS(MCTS):
    """
    From Moerland et al. (2022) — FRAP
    Introduces model-based rollout integration:
      - Uses both learned value estimates and simulated rollouts.
      - Demonstrates how planning can inform value-based learning and vice versa.
    """

    def __init__(self, game, iterations=1000, exploration=math.sqrt(2.0),
                 value_fn=None, alpha=0.5):
        super().__init__(game, iterations, exploration)
        self.value_fn = value_fn  # e.g., a learned value function approximator
        self.alpha = alpha  # mixing coefficient between model and learned value

    def _simulate(self, state):
        if self.value_fn:
            model_value = self.value_fn(state)
        else:
            model_value = 0.0

        rollout_value = super()._simulate(state)
        return self.alpha * rollout_value + (1 - self.alpha) * model_value


class PolicyMCTS(MCTS):
    """
    From IJCAI 2019 — MCTS for Policy Optimization
    Integrates a policy prior (softmax exploration) during expansion.
    """

    def __init__(self, game, policy_fn=None, temperature=1.0, **kwargs):
        super().__init__(game, **kwargs)
        self.policy_fn = policy_fn
        self.temperature = temperature

    def _select_weighted_action(self, state, actions):
        if self.policy_fn:
            probs = self.policy_fn(state, actions)
        else:
            probs = [1.0 / len(actions)] * len(actions)
        z = sum(math.exp(p / self.temperature) for p in probs)
        weights = [math.exp(p / self.temperature) / z for p in probs]
        return random.choices(actions, weights=weights, k=1)[0]

    def choose(self, root_state):
        root = MCTSNode(deepcopy(root_state), None, -self.game.current_player(root_state))
        root.untried_actions = self.game.legal_actions(root_state)
        for _ in range(self.iterations):
            node, state = root, deepcopy(root_state)
            while not node.untried_actions and node.children:
                node = node.uct_select_child(self.exploration)
                state = self.game.next_state(state, self._action_to_child(node, state))
            if node.untried_actions:
                action = self._select_weighted_action(state, node.untried_actions)
                node.untried_actions.remove(action)
                state = self.game.next_state(state, action)
                child = MCTSNode(deepcopy(state), node, -self.game.current_player(state))
                child.untried_actions = self.game.legal_actions(state)
                node.children.append(child)
                node = child
            result = self._simulate(state)
            self._backpropagate(node, result)
        best_child = max(root.children, key=lambda c: c.visits)
        return self._action_to_child(best_child, root_state)


class HierarchicalMCTS(MCTS):
    """
    From PRM-RL (Faust et al. 2017)
    Introduces a two-level MCTS:
      - High-level planner chooses macro-actions / waypoints.
      - Low-level planner executes MCTS locally to navigate there.
    """

    def __init__(self, game, macro_generator, iterations=500, sub_iterations=100):
        super().__init__(game, iterations)
        self.macro_generator = macro_generator
        self.sub_iterations = sub_iterations

    def choose(self, root_state):
        macros = self.macro_generator(root_state)
        best_macro, best_value = None, -float("inf")
        for macro in macros:
            subgame = deepcopy(self.game)
            local_mcts = MCTS(subgame, iterations=self.sub_iterations)
            sub_value = 0.0
            for _ in range(3):
                a = local_mcts.choose(root_state)
                state = subgame.next_state(root_state, a)
                sub_value += subgame.game_result(state)
            if sub_value > best_value:
                best_value, best_macro = sub_value, macro
        return best_macro


# --- Backup Tree (Lu et al., 2019) ---
class BackupTreeMCTS(MCTS):
    def backup(self, node, reward):
        # Back up all descendants on the optimal path
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        for depth, n in enumerate(reversed(path)):
            n.visits += 1
            n.value += (self.gamma ** depth) * reward


# --- Monte Carlo Tree Search for Policy Optimization (IJCAI 2019) ---
class MCTSPolicyOpt(MCTS):
    def backup(self, node, reward):
        # Combine planning + policy update idea
        while node:
            node.visits += 1
            advantage = reward - node.value / (node.visits + 1e-8)
            node.value += 0.1 * advantage  # gradient-style step
            node = node.parent


# --- RePReL (Relational Planning + RL) ---
class RePReLMCTS(MCTS):
    def expand(self, node):
        # Example: Relational abstraction (group similar actions)
        actions = self.env.get_actions(node.state)
        abstracted_actions = self.abstract_actions(actions)
        for a in abstracted_actions:
            next_state, reward, done = self.env.step(node.state, a)
            node.children.append(Node(next_state, parent=node))
        return random.choice(node.children) if node.children else node

    def abstract_actions(self, actions):
        # Placeholder for actual relational grouping
        return list(set(actions))


# --- PRM-RL (Faust et al., 2017) ---
class PRMRLMCTS(MCTS):
    def run(self, root_state):
        # Build PRM-style roadmap first
        roadmap = self.build_roadmap(root_state)
        best_path = self.plan_via_prm(roadmap)
        return best_path[-1] if best_path else root_state

    def build_roadmap(self, start):
        # Dummy probabilistic roadmap builder
        return [start] + [self.env.random_state() for _ in range(5)]

    def plan_via_prm(self, roadmap):
        # Example of RL-based local navigation between PRM nodes
        path = []
        for i in range(len(roadmap) - 1):
            path.append(roadmap[i + 1])
        return path


class RelationalMCTS(MCTS):
    """
    From RePReL — Integrating Relational Planning and RL
    Adds symbolic abstraction: states are mapped to equivalence classes via
    a relational abstraction function, encouraging generalization.
    """

    def __init__(self, game, abstraction_fn=None, **kwargs):
        super().__init__(game, **kwargs)
        self.abstraction_fn = abstraction_fn or (lambda s: s)

    def _simulate(self, state):
        abstract_state = self.abstraction_fn(state)
        return super()._simulate(abstract_state)
