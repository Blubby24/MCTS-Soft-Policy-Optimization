# Related Work

This section provides an in-depth, technical review of some of the core reading I completed for this research project. 
The paper I began with was How to Combine Tree-Search Methods in Reinforcement Learning which shaped the direction my work flowed.


---

## How to Combine Tree-Search Methods in Reinforcement Learning

The authors look at the idea of a Finite-horizon lookahead (run a tree search to depth `h`, evaluate leaves with current value estimates, and use the root decision or a root-backup to update policy/value). This idea is widely used and often effective, but the standard implementation pattern backing up only leaf returns and using only the root decision for policy updates does not define a contractive operator on value functions,
and therefore repeated application can fail to converge. The authors formalize tree-search-based lookahead as an operator on value functions, expose the non-contractivity of the naïve leaf-backup operator, and propose a minimal modification
propagate the return of the best path found in the tree to descendants near the root which restores contraction with factor gamma^h (where $\gamma$ is discount and h the depth). 
The main insight here is: deeper lookahead yields exponentially stronger contraction but only if you back up the best-path return appropriately.  


The paper uses the standard MDP notation: states $s$, actions $a$, transition kernel $P(s' \mid s, a)$, reward $r(s, a)$, discount $\gamma$. Let $V$ be a value function (a bounded real function on states). The usual Bellman optimality operator $T$ is the familiar  

$$
(T V)(s) = \max_a \left[ r(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s, a)} V(s') \right].
$$

Where a $h$-step lookahead (tree search of depth $h$) induces an operator that for each root state $s$ computes the maximum over action sequences $a_0, \dots, a_{h-1}$ of the $h$-step cumulative discounted reward plus the bootstrapped $V$ at the depth-h state $s_h$:

$$
T^{(h)} V(s) = \max_{a_0, \dots, a_{h-1}} \mathbb{E} \Big[ \sum_{t=0}^{h-1} \gamma^t r_t + \gamma^h V(s_h) \Big].
$$

Operationally, a planner expands a tree to depth $h$, evaluates leaves using $V$, and backs leaf values to the root using max/expectation operators. But the authors point out that if $V$ is approximate or noisy (as in function approximation), repeated application of this $T^{(h)}$-style operator need not be a contraction in sup-norm, so fixed-point convergence guarantees vanish; in other words, the iterative scheme  

$$
V_{k+1} = T^{(h)} V_k
$$  

may not converge to a single fixed point even in finite MDPs. The paper constructs simple counterexamples and explains how information loss near the root (only using the leaf returns to update the root) breaks the contraction property.  

The remedy the authors propose is deceptively simple and easy to implement: after the $h$-step search, identify the path (sequence of actions) from the root to some leaf that achieves the maximal estimated return under the current $V$-leaf evaluations; call the return along that path $R_{\text{best}}(s)$ instead of only using leaf-to-root backups in the usual way. The algorithm also propagates $R_{\text{best}}(s)$ down to the root’s descendants (the nodes that are direct children or within a small radius under the root) 
Basically assign the best-path return as an alternative estimate for those interior nodes. The resulting mapping, denoted in the paper as $\hat{T}^{(h)}$ (here, $\hat{T}$ stands for the enhanced/contracting operator), satisfies the key contraction bound  

$$
\| \hat{T}^{(h)} V_1 - \hat{T}^{(h)} V_2 \|_\infty \le \gamma^h \, \| V_1 - V_2 \|_\infty.
$$

This inequality is the core theorem: the enhanced lookahead operator is a $\gamma^h$-contraction in sup-norm, so repeated application converges geometrically to a unique fixed point, with rate controlled by $\gamma^h$. The intuition is that by copying the best-path return downward, you eliminate a source of destructive amplification of the estimation error: any difference in $V_1$ and $V_2$ at leaves gets attenuated by a factor $\gamma^h$ when mapped to the root and its close descendants. The full paper provides the formal operator definitions and the short, elementary proof of this contraction inequality.  


To connect this operator-level result to algorithm design, the authors define a notion they call multiple-step greedy consistency. Roughly speaking, multiple-step greedy consistency is a compatibility condition requiring that the $h$-step greedy policy produced by planning is consistent with the backing value $V$ (i.e., the planner’s choices at the root align with the greedy choices implied by the approximated $V$ values used at leaves). When this consistency condition holds (or approximately holds within bounded noise), the contraction guarantee ensures that iteratively applying the contracting operator yields monotone improvements and convergence to an optimal value within the approximation error bounds. The paper formalizes the condition and shows how bounded deviations from it produce bounded asymptotic errors this is the exact statement that lets one reason about noisy tree search and noisy value estimates.  


Recognizing that practical planners and approximators are noisy, the authors analyze two algorithmic instantiations of the contracting backup idea under stochastic noise models: one is a deterministic-style update with bounded additive noise in the estimates, and the other is a sampling-style stochastic algorithm modeling finite-sample tree search noise (e.g., Monte Carlo sampling in MCTS). For each instantiation they provide finite-sample or finite-iteration bounds showing how the error depends on $h$, $\gamma$, and the amplitude of noise. The bounds have the intuitive form that deeper lookahead (larger $h$) multiplies the contraction advantage ($\gamma^h$) but at the cost of requiring more samples to get accurate search estimates (because the tree branching grows exponentially with $h$). Concretely, the error of repeated application of the noisy contracting operator decays geometrically at rate $\gamma^h$ up to an additive bias term that scales with the magnitude of worst-case noise; thus, if tree-search noise and value-estimator noise are controlled, deeper lookahead yields faster convergence in practice, but the constants can be large when noise is high. The formal statements in the paper quantify these trade-offs and give explicit sample-complexity style inequalities for the stochastic instantiation.  


The paper also contains algorithmic pseudocode (readily translatable to typical MCTS frameworks) showing exactly where to add the contracting backup step. In routine MCTS the steps at a root are: selection (UCB down the tree), expansion (add new child nodes), simulation/rollout (estimate leaf returns), and backpropagation (propagate rollout returns up the tree, updating visit counts and value estimates). The contracting backup modification only inserts an extra backpropagation of the single best-path return $R_{\text{best}}$ to the root’s immediate subtree: after the normal rollouts/backups finish, compute which leaf path produced the largest estimated return; then, for the nodes along that best path and for the other nodes in the root’s local neighborhood, update their stored value estimates (or add an additional pseudo-update) with $R_{\text{best}}$ (usually applying the same visit-count-normalized averaging more sophisticated code uses). Implementationally this is cheap: you only need to store the best-seen path return during the search and do a second pass of localized updates. The authors highlight that it is not necessary to alter the selection or expansion heuristics much: the contracting backup simply changes the way leaf information influences parent estimates.  


From a systems perspective there are several practical notes and tradeoffs the paper emphasizes. The $\gamma^h$ contraction factor looks attractive (exponentially small contraction factor as $h$ grows), but $h$ cannot be increased arbitrarily because tree search cost (time and samples) grows quickly with depth and branching factor. In sampling-based tree-search (MCTS-style) you need sufficient rollouts per node to accurately identify the best path; otherwise the best-path identification can be wrong and the contracting backup will propagate biased information. Thus there is a bias-variance / compute tradeoff: deeper $h$ increases stability per operator application but typically requires more computation to ensure the search produces an accurate $R_{\text{best}}$. Their stochastic finite-sample bounds make this tradeoff explicit: the additive error term in the bound decreases with more search samples at fixed $h$, and the multiplicative contraction $\gamma^h$ reduces propagation of residual leaf-value estimation error. The takeaway is practical: use contracting backups when you can afford reasonably precise search (or when the value approximator is noisy), and tune $h$ and search effort jointly.


# Monte‑Carlo Tree Search for Policy Optimization (MCTSPO)

Monte-Carlo Tree Search for Policy Optimization (MCTSPO) introduces a novel approach to reinforcement learning by reformulating policy optimization as a deterministic Markov Decision Process (MDP) in the space of policy parameters rather than environment states. In this formulation, a state $s$ corresponds to a complete set of policy parameters, typically the weights of a neural network, while an action $a$ corresponds to a mutation of the parameter vector, represented as a pair $[{\rm seed}, v]$ where the seed determines a stochastic direction $\delta$ and $v$ is a scalar magnitude. Transitions are deterministic and given by $s_{t+1} = s_t + v \cdot \delta$. The reward of a transition is defined as the improvement in expected return induced by the mutated policy, i.e., $R(s, a, s') = \eta_{\rm env}(s') - \eta_{\rm env}(s)$, where $\eta_{\rm env}(s)$ is the expected return of policy $s$ in the original task environment. The objective becomes to find a sequence of mutation actions that maximizes cumulative reward in this parameter-space MDP, $\max_{a_0, \dots, a_{H-1}} \sum_{t=0}^{H-1} R(s_t, a_t, s_{t+1})$, effectively performing lookahead over sequences of policy updates.

To solve this MDP efficiently, the authors adapt Monte-Carlo Tree Search (MCTS) to the parameter space, using a tree where each node corresponds to a policy parameter vector and edges correspond to mutation actions. The search is guided by an Upper Confidence Bound (UCB) rule adapted for deterministic transitions, selecting child nodes $i$ with maximum R̄ᵢ + c × √( ln(N_parent) / Nᵢ ), where $\bar{R}_i$ is the accumulated return of node $i$, $N_i$ its visit count, and $c$ a tunable exploration constant. Since the action space is continuous and high-dimensional, they employ progressive widening, incrementally adding new child mutations only as the parent node’s visit count increases, thereby controlling branching factor growth. Rollouts are realized by deploying the policy at the leaf node in the original environment and measuring its return, which is propagated up the tree deterministically using max-backup rather than averaging, reflecting the deterministic nature of transitions.

A key technical contribution is the adaptation of MCTS to high-dimensional, continuous parameter spaces. The progressive widening strategy ensures that at low visit counts only a few mutations are considered, preventing exponential branching, while more promising nodes are gradually expanded as more computational budget is spent. Additionally, the deterministic backup operator propagates the maximum observed cumulative return from child to parent nodes, in contrast to standard MCTS which averages stochastic returns; this leverages the deterministic structure of the parameter-space MDP and ensures that exploration focuses on sequences leading to the largest improvement in policy performance. The evaluation of each policy is computationally expensive, but by performing multiple mutations and deep tree search, the method effectively performs multi-step lookahead in the space of policy parameters, analogous to the $h$-step lookahead operators studied in value-based MDPs, but without relying on a learned value function $V(s)$.

The reward structure of MCTSPO is inherently sparse and non-stationary: early mutations may produce policies that perform poorly, resulting in negative or near-zero rewards, while later mutations may yield significant improvements. The algorithm handles this by maintaining cumulative returns along each path and selecting actions that maximize these multi-step improvements. Formally, for a path $s_0 \to s_1 \to \dots \to s_H$, the backed-up return at the root is $R_{\rm path} = \sum_{t=0}^{H-1} R(s_t, a_t, s_{t+1})$, and the node values in the tree are updated according to the maximum observed $R_{\rm path}$ among all children, implementing a form of best-path propagation similar in spirit to contracting backups in value-based tree search.

Empirically, MCTSPO is evaluated on continuous-control benchmark tasks including classic control problems and robotics simulators. The method outperforms both gradient-based policy optimization algorithms, such as TRPO, and population-based evolutionary strategies with safe mutation. The advantage arises primarily in tasks with deceptive or sparse rewards, where gradient signals are weak or misleading. In these environments, multi-step lookahead over sequences of parameter mutations allows MCTSPO to discover higher-reward regions of the policy space more efficiently. Notably, the method scales with tree depth $H$ and mutation magnitude $v$, trading off computational cost for more accurate exploration: deeper trees ($H$ larger) yield more informed policy updates at the expense of increased rollouts, while larger mutation magnitudes allow broader exploration but risk overshooting promising regions.

The algorithm can be understood as performing a structured, deterministic search in the neighborhood of the current policy parameters, where the tree encodes sequences of parameter-space perturbations and the deterministic backup ensures that only the most promising trajectories are reinforced. In contrast to standard evolutionary strategies that treat parameter mutations independently, MCTSPO effectively builds a multi-step plan over parameter sequences, akin to performing lookahead in classical MDPs. The combination of progressive widening, deterministic backup, and UCB-guided selection enables the algorithm to navigate high-dimensional continuous spaces efficiently, yielding substantial performance improvements on tasks with challenging reward landscapes.

The strengths of MCTSPO include its ability to perform multi-step lookahead in the high-dimensional parameter space, which allows it to efficiently discover promising policies even when rewards are sparse or deceptive. The deterministic backup ensures that only the most promising trajectories influence policy updates, avoiding averaging that can dilute useful signals. Progressive widening allows the method to scale to continuous action spaces without an exponential blow-up in branching factor. Weaknesses include the high computational cost of evaluating multiple full-environment rollouts, the sensitivity to hyperparameters such as tree depth $H$ and mutation magnitude $v$, and the difficulty of scaling to very high-dimensional neural networks due to the combinatorial explosion of possible mutation sequences.

# Soft Policy Optimization (SPO)

Soft Policy Optimization (SPO) introduces a soft-reinforcement-learning approach for sequence-model policies, such as language or code models. Standard RL fine-tuning of these models, typically using on-policy methods like PPO, is sample-inefficient and cannot easily reuse arbitrary past trajectories. SPO overcomes these limitations by supporting offline and online data, avoiding a separate value network, and learning a soft action-value function $Q_\theta$ in a memory-efficient way.  

In SPO, given a prompt $x$, the policy produces a token sequence $a = a_1, \dots, a_T$, and receives a reward $r(x, a)$, typically only at the end of the sequence. There is no discounting or intermediate reward, and the environment is deterministic. The core idea is the cumulative Q-parameterization, where $Q_\theta$ encodes the “surprising but good” nature of token sequences relative to a reference model's log-probabilities. This parameterization ensures soft Bellman-consistency and path-consistency by construction, except at terminal tokens where the actual reward is observed.  

SPO derives off-policy loss objectives. Terminal Q-regression updates the final $Q_\theta$ toward the observed reward using standard loss functions. For sequences generated by older policies, importance weighting corrects distribution shifts before applying updates. This allows SPO to learn from offline trajectories from prior runs or human data, improving sample efficiency and flexibility.  

The method maintains cumulative returns along sequences and propagates updates only at terminal tokens. Formally, for a sequence $a_0, \dots, a_{T-1}$, the backed-up return is $R_{\rm path} = \sum_{t=0}^{T-1} r(x, a_t)$, and the token-level Q-values are adjusted according to this cumulative reward. Because Q-values are defined in terms of policy and reference log-probabilities, no separate value function is needed, and updates are memory-efficient.  

Empirically, SPO achieves higher success rates on programming benchmarks compared to PPO, benefiting from multi-step trajectory learning and the ability to leverage offline data. The method scales with the sequence length $T$ and the quality of the reference model; longer sequences allow richer off-policy information, and a strong reference model stabilizes Q-values. The approach effectively performs structured soft RL without requiring explicit value function training or environment stochasticity.  

Strengths of SPO include support for off-policy and offline data, memory efficiency through cumulative Q-parameterization, simplicity and scalability of the update rules, better policy diversity due to soft RL, and empirical performance improvements over standard PPO baselines. Weaknesses include reliance on simplified reward setups (deterministic, terminal-only, sparse), dependence on reference model quality for Q-value fidelity, potential instability in importance weighting when using old trajectories, limited theoretical guarantees in realistic stochastic or partially observed settings, and challenges when rewards are delayed, sparse, or noisy.  

Compared to MCTSPO (2019), which frames policy optimization as a deterministic MDP in parameter space and performs Monte-Carlo Tree Search over mutation sequences $s_{t+1} = s_t + v \cdot \delta$ with deterministic backups and multi-step lookahead, SPO focuses on discrete-sequence generation. MCTSPO is effective for low-dimensional continuous control tasks but is very sample- and computation-intensive for large parameter spaces. SPO avoids explicit search over parameters, supports off-policy trajectory reuse, and is memory-efficient, making it suitable for large sequence models. While MCTSPO relies on environment rollouts and tree search to explore parameter space, SPO performs soft RL in token space, with theoretical consistency guaranteed under idealized assumptions but practical performance empirically validated in deterministic sequence-generation tasks.  

SPO demonstrates that off-policy soft RL without a separate value network can scale to modern sequence models, providing efficient, stable, and diverse policy learning. Its key technical contribution is the cumulative Q-parameterization combined with off-policy updates, which enables multi-step learning from both online and offline sequences while maintaining a single unified model for policy and value representation.

# Methods

While prior work in reinforcement learning and planning has explored model-free policy optimization, tree-search planning, and trajectory-level soft RL independently, there has been limited exploration of closed-loop integration between environment-level planning and trajectory-based policy learning. My work addresses this gap by combining Soft Policy Optimization (SPO) with Monte Carlo Tree Search (MCTS) in a feedback loop.

The motivation arises here from the complementary strengths of the two approaches: MCTS excels at discovering high-reward action sequences in sparse or deceptive environments, while SPO can absorb off-policy trajectories without requiring a separate value network and can provide a memory-efficient, soft Q-based policy representation. By training a soft policy $\pi_\theta$ from MCTS-generated trajectories and using it as a prior for subsequent search, I can iteratively refine the prior and focus tree expansions on promising branches. This approach effectively alternates search, data collection, and off-policy soft RL updates, leveraging the multi-step planning advantages of MCTS with the trajectory-level learning capabilities of SPO.

The environment for testing is a deterministic grid maze with sparse rewards: the agent receives a reward of $+1$ upon reaching the exit and a small negative or zero reward per step ($-0.01$ or $0$) to encourage efficiency. Procedurally generated mazes are used to evaluate generalization, similar to Minigrid-style environments or a custom gridworld. Each episode starts with the agent at a fixed or random initial location and terminates when the exit is reached or a maximum number of steps is exceeded.

The experimental procedure is as follows. Initialize the prior policy $\pi_\theta$ randomly or with a simple pretrained policy. For each iteration of training, run MCTS from the current root state using $\pi_\theta$ as the prior for node selection. Collect the top-K trajectories from the search tree, including state sequences, actions, cumulative returns, and visit counts. Store these trajectories in a dataset $D$ and perform a soft off-policy update of $\pi_\theta$ using the SPO objective, which increases the probability of high-return sequences while maintaining entropy regularization to prevent collapse. Formally, for a trajectory $\tau = (s_0, a_0, \dots, s_H)$ with cumulative return $R_{\rm path}$, the update adjusts $Q_\theta(s_t, a_t)$ toward $R_{\rm path}$ for all $t$, incorporating importance weighting if the trajectory comes from an older policy.

The procedure can be expressed as pseudocode:

```text
initialize πθ

for iter = 1..N:
    D = []
    for episode = 1..M:
        tree = MCTS(root=env_state, prior=πθ, budget=B, depth=h)
        topK = extract_topK(tree)   # top-K trajectories by return
        D.add(topK)                 # store (states, actions, returns, visits)
    θ ← SPO_update(θ, D)            # soft off-policy update
```

Key hyperparameters include tree depth $h$, number of MCTS simulations per root $B$, top-K trajectories per episode, number of SPO update steps per iteration, learning rate, and entropy coefficient. Typical values might be h = 10–40 steps depending on maze size, B = 50–300 for training and 500+ for evaluation, top-K = 5–20, SPO steps = 1–5 with learning rate tuned around $1\mathrm{e}{-4}$, and entropy coefficient tuned to avoid premature policy collapse.

MCTS generates high-quality trajectories in each maze episode, and SPO distills this information into a soft prior policy $\pi_\theta$. This prior biases subsequent MCTS expansions toward promising branches, effectively creating a positive feedback loop: better search informs better policy, which then focuses future search more efficiently. The approach naturally handles sparse rewards because the prior helps guide exploration toward trajectories with nonzero returns, while the soft Q-values and entropy regularization prevent premature convergence to suboptimal paths.

Conceptually, this framework unites multi-step planning and trajectory-level learning. MCTS provides structured, deterministic search over sequences, similar to the best-path backups in MCTSPO, while SPO aggregates cumulative returns from these sequences, enabling off-policy updates and generalization across procedurally generated mazes. This combination balances exploration, sample efficiency, and policy generalization, making it well suited to evaluate iterative search and soft policy learning in environments where sparse rewards and multi-step reasoning are essential.

## Implementation

Practically the core idea is that a learned policy network guides Monte Carlo Tree Search (MCTS), while MCTS outcomes are used to improve the policy. This combination allows the agent to navigate mazes efficiently, correcting suboptimal policy actions through planning and reducing unnecessary exploration in the tree search. At the heart of the system is the policy network, implemented as a convolutional neural network (PolicyNetwork). The network takes as input a 3-channel representation of the maze, encoding the agent position, the goal position, and obstacles. Three convolutional layers with ReLU activations extract spatial features from this input. The outputs are flattened and passed through fully connected layers to produce logits, which are converted to a probability distribution over the four actions (up, down, left, right) using a softmax. This network is capable of providing action probabilities for individual states, which serve two purposes in the implementation: selecting actions in policy-only mode and providing priors for MCTS node expansion.

MCTS is implemented in the MCTS class, with nodes represented by MCTSNode. Each node corresponds to a maze state, and edges correspond to actions. A search begins at the root node representing the current state. Child nodes are selected using the PUCT formula, which balances the mean value of the node, the prior from the policy network, and visit counts of the parent and child. In the code, _select_child implements this selection, ensuring a balance between exploitation and exploration. When a leaf node is reached, _expand adds children for all possible actions, initializing action priors using the policy network. Rollouts are performed in _simulate, where actions are sampled stochastically according to the policy probabilities until a terminal state or a maximum depth is reached. The discounted cumulative reward from this simulation is backpropagated up the tree using _backpropagate, updating visit counts and value sums along the path. The best action at the root is selected using get_best_action, based on the most visited child. Trajectories are extracted using extract_trajectories, which performs a depth-first traversal of the tree and selects the top-K paths with the highest cumulative returns.

The SPO trainer (SPOTrainer) updates the policy network using trajectories extracted from MCTS. Each trajectory contains states, actions, rewards, and discounted returns. The trainer computes importance weights for each state-action pair by comparing returns to a baseline stored in an exponential moving average (baseline_ema) and scales these weights using a temperature parameter beta. The loss function combines a weighted negative log-likelihood with entropy regularization, ensuring that high-return trajectories influence policy updates while preserving stochasticity. Updates are applied over multiple SPO steps per iteration, gradually improving the policy while avoiding premature collapse into a deterministic function. This allows the policy to maintain exploration while reinforcing actions that lead to successful trajectories.

The training loop, implemented in train_spo_mcts, alternates between data collection and policy update phases. During data collection, the environment is explored using MCTS guided by the current policy, and trajectories are extracted from the tree. In the policy update phase, the SPO trainer uses these trajectories to update the network parameters. Metrics such as success rate, average steps, number of trajectories, loss, and entropy are logged to monitor training progress. This closed-loop interaction enables the policy to improve over iterations while simultaneously improving the efficiency of MCTS, as better priors focus the search on promising actions.

# Results, Analysis and Discussion

## Experimental Setup

The experiments were conducted in a square grid maze environment (`MazeEnv`) of size 7×7 with an obstacle density of 0.2, meaning 20% of the cells were blocked. Each step incurred a small penalty of -0.01, and episodes were allowed to continue until reaching either the goal or the environment-defined maximum number of steps. To ensure reproducibility, a fixed seed of 42 was used across all experiments, and the same maze configurations were applied to every method. Evaluation was performed over 15 episodes per method.

Three different methods were compared. The first, a random policy, chose actions uniformly at random and served as a lower bound on performance. Metrics tracked for this baseline included success rate and average steps, both across all episodes and for successful episodes only. The second method, pure MCTS, used Monte Carlo Tree Search with uniform priors and a budget of 100 simulations per step. Its purpose was to measure the performance of planning without a learned policy. Metrics included success rate, average steps, and average MCTS simulations per episode. Finally, SPO-MCTS, combined a learned policy network with MCTS. The policy network was trained using Soft Policy Optimization (SPO) and updated based on trajectories extracted from MCTS searches. Training consisted of 5 iterations with 5 episodes per iteration, and an MCTS budget of 30 simulations per step during training. Evaluation of SPO-MCTS was performed in two modes: policy-only, in which actions were selected greedily according to the learned policy, and policy+MCTS, in which MCTS guided by the policy network selected actions with a budget of 1000 simulations per episode.

I tracked several metrics across all experiments. Success rate the proportion of episodes in which the agent reached the goal. Average steps indicated the mean number of steps per episode, both over all episodes and for successful episodes only. Average MCTS simulations quantified the search effort per episode, and during training, additional metrics such as success rate per iteration, number of extracted trajectories, policy loss, and policy entropy were recorded to monitor learning progress.

The SPO-MCTS implementation integrates learning and planning in a closed loop, where a policy network guides MCTS while also being updated using the results of the search. The policy network is a convolutional neural network that maps the current maze state to a probability distribution over the four possible actions: up, down, left, and right. It has three convolutional layers with ReLU activations to extract spatial features from the maze state, with input channels representing the agent position, the goal position, and obstacles. These convolutional outputs are flattened and passed through fully connected layers to produce logits, which are transformed with a softmax to give a valid probability distribution over actions. The network provides methods to retrieve action probabilities for individual states, which are used both for policy-only action selection and to provide priors for MCTS during tree search.

MCTS is implemented as a tree search where nodes correspond to states and edges correspond to actions. Each simulation starts at the root node, and the algorithm selects child nodes according to the PUCT formula, which balances the value of a node with the prior probability provided by the policy network and the visit counts of the parent and child nodes. When a leaf node is reached, all possible actions are expanded as children, and the action priors are initialized using the policy network. From the expanded node, a rollout is performed by selecting actions stochastically according to the policy probabilities until either a terminal state is reached or a maximum depth is exceeded. The discounted cumulative reward from this simulation is then backpropagated up the tree, updating the visit counts and value sums of all nodes along the path. The best action at the root is selected based on the highest visit count, although a full probability distribution over actions can also be derived from the visit statistics if needed.

Trajectories from MCTS searches are extracted to train the policy network. For each root search, a depth-first traversal collects the top-k trajectories, prioritizing paths with higher cumulative returns. Each trajectory includes the states, actions, rewards, discounted return, and length of the path. These trajectories are collected across all episodes in a training iteration and serve as the dataset for policy updates.

Policy updates are performed using soft policy optimization. The policy network is updated by maximizing the expected return of the collected trajectories while maintaining some entropy in the action distribution. Each update uses multiple SPO steps per iteration, gradually adjusting the network to reinforce actions that lead to high-return paths while avoiding premature convergence to a deterministic policy. The training loop consists of two main phases in each iteration: a data collection phase, where MCTS guided by the current policy explores the environment and generates trajectories, and a policy update phase, where the collected trajectories are used to improve the network. Metrics such as success rate, average steps, number of trajectories, loss, and entropy are logged for monitoring progress.

After training, I evaluate the policy network in two modes. In policy-only mode, actions are selected greedily according to the learned probabilities to assess the strength of the policy itself. In policy+MCTS mode, the trained policy is used as a prior for MCTS, which selects actions with a higher simulation budget to combine the efficiency of the policy with the corrective power of tree search. This setup allows the system to reduce unnecessary exploration while retaining the ability to correct suboptimal decisions, resulting in more efficient and successful navigation through the maze.



### Results

###  Baseline Performance

#### Random Policy
- **Success Rate**: 20% (3/15)  
- **Average Steps (all episodes)**: 87.3  
- **Average Steps (successful episodes)**: 44.3  

#### Pure MCTS (Budget=100)
- **Success Rate**: 93.3% (14/15)  
- **Average Steps (all episodes)**: 20.7  
- **Average Steps (successful episodes)**: 15.1  
- **Average MCTS Sims per Episode**: 2067  

---

### SPO-MCTS Training Progress

| Iteration | Success Rate | Avg Steps | Loss     | Entropy | Trajectories |
|-----------|--------------|-----------|----------|---------|--------------|
| 1/5       | 80% (4/5)    | 44.2      | -0.0126 | 1.386   | 252          |
| 5/5       | 100% (5/5)   | 16.6      | -0.0114 | 1.378   | 98           |

- **Observation**: Success rate improves rapidly over 5 iterations, and average steps decrease, indicating learning of more efficient policies.  
- Loss and entropy remain stable, showing stable policy network updates.

---

### SPO-MCTS Evaluation

#### Policy-only
- **Success Rate**: 0%  
- **Average Steps**: 98.0  

#### Policy + MCTS
- **Success Rate**: 100%  
- **Average Steps**: 14.6  
- **Average MCTS Sims**: 1460  


---

### Final Comparison Table

| Method                     | Success Rate | Avg Steps | MCTS Sims |
|-----------------------------|--------------|-----------|-----------|
| Random Policy               | 20.0%        | 87.3      | N/A       |
| Pure MCTS                   | 93.3%        | 20.7      | 2067      |
| SPO-MCTS (Policy Only)      | 0.0%         | 98.0      | 0         |
| SPO-MCTS (Policy + MCTS)    | 100.0%       | 14.6      | 1460      |

---

## Analysis

The results of my experiments reveal several important insights about the performance and limitations of SPO-MCTS in comparison to the baseline methods. When evaluated against the random policy, SPO-MCTS with MCTS shows a dramatic improvement in success rate, increasing from 20% to 100%. This demonstrates that even with a relatively small training budget and few iterations, the learned policy provides a meaningful prior that significantly guides the search process, enabling the agent to navigate the maze efficiently. The high improvement highlights the ability of SPO-MCTS to leverage structured exploration over purely random action selection, which is highly ineffective in sparse-reward environments like this maze.

Interestingly, when evaluating the learned policy alone without MCTS, SPO-MCTS fails completely, achieving a 0% success rate. This indicates that while the policy network captures some information about good actions during training, it is insufficient on its own to reliably reach the goal. The failure of the policy-only mode can be attributed to the sparse rewards and limited number of training iterations. Since the agent receives minimal feedback when it fails to reach the goal, the policy network does not encounter enough successful trajectories to generalize across the state space. This highlights a key limitation of purely policy-based approaches in environments with delayed or infrequent rewards.

The combination of the learned policy with MCTS, however, not only recovers the performance but exceeds that of pure MCTS, achieving a 100% success rate with fewer simulations per episode (1460 vs 2067). This demonstrates a clear synergy between learning and planning: the policy network effectively biases the MCTS search toward promising actions, reducing wasted simulations in unproductive regions of the search tree. As a result, SPO-MCTS achieves comparable or better success while being more computationally efficient. This efficiency gain is critical in larger or more complex environments, where pure MCTS could become prohibitively expensive.

From a methodological standpoint, these results suggest that the SPO updates successfully integrate trajectory information into the policy network, enabling it to encode useful priors for guiding search. The reduction in training trajectories over iterations, accompanied by increasing success and decreasing step count, indicates that the policy network is learning to focus exploration more effectively and that MCTS can exploit these priors to refine decision-making. However, the zero success of the policy-only mode also emphasizes that additional training iterations, more diverse environments, or enhanced reward shaping would be necessary for the policy to perform independently.

In terms of broader implications, this experiment supports the idea that combining learning and planning can substantially improve both performance and efficiency in sequential decision-making tasks. The learned policy acts as a form of informed heuristic that reduces the burden on planning algorithms, while planning compensates for the policy's imperfections. This approach could scale to larger mazes or more complex planning problems where relying solely on either policy learning or tree search would be insufficient. It also opens avenues for further research, such as adaptive MCTS budgets based on policy confidence, curriculum learning to improve policy generalization, and integration with denser reward signals to enable stronger policy-only performance. Overall, SPO-MCTS demonstrates a compelling framework for harnessing the complementary strengths of learning and planning in challenging environments.


## Discussion

The experimental results suggest that SPO-MCTS successfully integrates learned policy priors with Monte Carlo Tree Search to create a more efficient and effective navigation agent in maze environments. By training a convolutional policy network using trajectories extracted from MCTS searches, I was able to provide the tree search with a prior that significantly reduced the number of unnecessary simulations while maintaining or slightly improving success rates compared to pure MCTS. The policy effectively acted as a guide, biasing exploration toward promising actions and thereby allowing MCTS to focus computational effort on evaluating critical decision points rather than uniformly exploring all possibilities. This led to a measurable improvement in search efficiency: the agent achieved comparable or better success rates with fewer average simulations per episode. These results indicate that even moderately trained policies can provide meaningful benefits when coupled with planning, particularly in structured environments like grid mazes where spatial regularities exist and can be exploited by convolutional networks.

The comparison between policy-only and policy+MCTS evaluation highlights the limitations of relying solely on learned policies in sparse-reward settings. In the policy-only mode, the agent often failed to reach the goal because the training process did not provide sufficient exploratory coverage of the state space. Sparse rewards meant that the policy network received weak and infrequent signals, which limited its ability to generalize across unseen maze configurations. This reinforces the importance of maintaining a planning component during evaluation: MCTS acts as a corrective mechanism that can identify and follow high-value trajectories even when the policy is imperfect. The combination of learning and planning thus offers robustness that neither approach could achieve independently. It also suggests that while the learned policy is beneficial for guiding search, additional techniques such as curriculum learning, reward shaping, or auxiliary exploration objectives may be necessary to improve the standalone performance of the policy network.

From a practical perspective, SPO-MCTS has several advantages and limitations. The reduction in required simulations makes it computationally attractive for real-time applications, such as robotic navigation or autonomous decision-making in complex but structured environments. By leveraging the policy network as a prior, I can allocate fewer resources to exhaustive planning, potentially allowing the agent to operate efficiently under strict computational budgets. Moreover, the approach is flexible: the MCTS component can compensate for policy errors, which is valuable in safety-critical applications where mistakes are costly. On the other hand, the experiments also reveal that the approach’s effectiveness is environment-dependent. The policy network is trained on relatively small mazes with fixed obstacle densities, and its generalization to larger or more dynamic environments is uncertain. In practice, scaling to larger mazes would require both longer training and careful tuning of MCTS parameters, such as simulation budget, maximum depth, and exploration constants, to maintain efficiency while avoiding combinatorial explosion in the search tree.

The results also highlight opportunities for methodological refinement. One limitation observed was the reliance on a fixed simulation budget during both training and evaluation. Adaptive simulation budgets, adjusted dynamically based on policy confidence or uncertainty estimates, could further improve efficiency by allocating computational effort only where the policy is less certain. Additionally, the sparse-reward setting exposed weaknesses in policy-only learning; future work could explore denser or shaped reward functions, curriculum-based maze progression, or auxiliary tasks to improve coverage of the state space during training. From an architectural standpoint, enhancements to the policy network, such as residual connections, attention mechanisms, or integration of recurrent components, could provide stronger representations of maze dynamics and improve trajectory prediction. Finally, the trajectory extraction process could be extended to include not only top-K high-return paths but also diverse or underexplored trajectories, which might provide richer training signals for the policy network and improve robustness in novel mazes.

Overall, the experiments confirm that integrating learning with planning is a highly effective strategy in navigation tasks. The learned policy provides a valuable inductive bias that focuses search toward promising actions, while MCTS ensures that mistakes are corrected and that the agent can handle sparse or deceptive reward structures. This combination yields a practical approach that balances computational efficiency with performance reliability. While policy-only learning remains insufficient in sparse-reward environments, SPO-MCTS demonstrates that combining planning with even moderately trained policies can outperform traditional search-based methods in both efficiency and success. 

# Bibliography

1. Ma, X., Driggs‑Campbell, K., Zhang, Z., & Kochenderfer, M. J. (2019). *Monte‑Carlo Tree Search for Policy Optimization*. In Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI 2019), pp. 3116‑3122. doi:10.24963/ijcai.2019/432  
   [arXiv preprint: arXiv:1912.10648] 

2. Efroni, Y., Dalal, G., Scherrer, B., & Mannor, S. (2018). *Multiple‑Step Greedy Policies in Online and Approximate Reinforcement Learning*. In Advances in Neural Information Processing Systems (NeurIPS 2018).
   [arXiv preprint: arXiv:1805.07956]

3. Efroni, Y., Dalal, G., Scherrer, B., & Mannor, S. (2019). *How to Combine Tree‑Search Methods in Reinforcement Learning*. arXiv:1809.01843. 

4. Moerland, T. M., Broekens, J., Plaat, A., & Jonker, C. M. (2022). *A Unifying Framework for Reinforcement Learning and Planning (FRAP)*. Frontiers in Artificial Intelligence, 5:908353. doi:10.3389/frai.2022.908353 

5. Bertsekas, D. P. (2011). *Approximate Policy Iteration: A Survey and Some New Methods*. Journal of Control Theory and Applications, 9(3), 310–335. doi:10.1007/s11768-011-1005-3 








