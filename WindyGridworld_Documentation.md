# Windy Gridworld Problem: Complete Mathematical Solution and Implementation Guide

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Mathematical Formulation](#mathematical-formulation)
3. [SARSA Algorithm Theory](#sarsa-algorithm-theory)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Code Walkthrough](#code-walkthrough)
6. [Results and Analysis](#results-and-analysis)
7. [Key Insights](#key-insights)

---

## Problem Overview

The **Windy Gridworld** is a classic reinforcement learning problem that demonstrates temporal difference learning in environments with stochastic transitions. It's particularly useful for understanding how agents can learn optimal policies when the environment has unpredictable elements (wind).

### Problem Setup
- **Grid**: 7×10 rectangular grid
- **Start Position**: (3, 0) - row 3, column 0
- **Goal Position**: (3, 7) - row 3, column 7
- **Wind Effect**: Columns have varying upward wind strength that affects movement
- **Actions**: 4 possible actions - Up, Down, Left, Right
- **Objective**: Find the shortest path from start to goal considering wind effects

### Wind Pattern
```
Column:  0  1  2  3  4  5  6  7  8  9
Wind:    0  0  0  1  1  1  2  2  1  0
```

---

## Mathematical Formulation

### 1. Markov Decision Process (MDP) Definition

The Windy Gridworld can be formulated as an MDP with the following components:

**State Space (S)**: 
- S = {(i,j) | 0 ≤ i < 7, 0 ≤ j < 10}
- |S| = 70 states
- Each state represents a position (row, column) in the grid

**Action Space (A)**:
- A = {0, 1, 2, 3} representing {Up, Down, Left, Right}
- |A| = 4 actions available from any state

**Transition Function (P)**:
The transition probability P(s'|s,a) is deterministic given the wind effect:

```
For action a from state s = (i,j):
1. Apply action: (i',j') = (i + Δi_a, j + Δj_a)
2. Apply wind: i'' = i' - wind[j]
3. Clip to boundaries: s' = (max(0, min(6, i'')), max(0, min(9, j')))
```

Where action effects are:
- Up (0): Δi = -1, Δj = 0
- Down (1): Δi = +1, Δj = 0  
- Left (2): Δi = 0, Δj = -1
- Right (3): Δi = 0, Δj = +1

**Reward Function (R)**:
```
R(s,a,s') = -1 for all transitions (living penalty)
```
This encourages the agent to find the shortest path.

**Discount Factor (γ)**:
- γ = 1.0 (undiscounted episodic task)

### 2. Value Functions

**State-Action Value Function (Q-function)**:
```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
```

Where G_t is the return:
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

**Optimal Q-function**:
```
Q*(s,a) = max_π Q^π(s,a)
```

**Bellman Equation for Q***:
```
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]
```

---

## SARSA Algorithm Theory

SARSA (State-Action-Reward-State-Action) is an **on-policy** temporal difference learning algorithm.

### 1. Algorithm Overview

SARSA learns the action-value function Q(s,a) for the policy it's following (ε-greedy in our case).

**Key Characteristics**:
- **On-policy**: Learns about the policy being followed
- **Temporal Difference**: Updates estimates based on other estimates
- **Model-free**: Doesn't require knowledge of transition probabilities

### 2. SARSA Update Rule

The core update equation is:
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Components**:
- **α**: Learning rate (0 < α ≤ 1)
- **γ**: Discount factor (0 ≤ γ ≤ 1)
- **TD Error**: δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
- **TD Target**: R_{t+1} + γQ(S_{t+1}, A_{t+1})

### 3. ε-Greedy Policy

The agent follows an ε-greedy policy for exploration:

```
π(a|s) = {
    1 - ε + ε/|A(s)|  if a = argmax_a Q(s,a)
    ε/|A(s)|          otherwise
}
```

**Probability of action selection**:
- **Exploitation**: Choose best action with probability (1-ε)
- **Exploration**: Choose random action with probability ε

### 4. Convergence Properties

SARSA converges to the optimal policy under these conditions:
1. All state-action pairs are visited infinitely often
2. Learning rate satisfies: Σ_t α_t = ∞ and Σ_t α_t² < ∞
3. ε decreases to 0 over time (for optimal convergence)

---

## Step-by-Step Implementation

### Step 1: Environment Setup

**Initialize the grid world**:
```python
class WindyGridworld:
    def __init__(self):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
```

**State transition logic**:
```python
def step(self, action):
    row, col = self.current_state
    
    # Apply action
    d_row, d_col = self.action_effects[action]
    new_row = row + d_row
    new_col = col + d_col
    
    # Apply wind (upward)
    wind_strength = self.wind[col]
    new_row -= wind_strength
    
    # Boundary clipping
    new_row = max(0, min(self.height - 1, new_row))
    new_col = max(0, min(self.width - 1, new_col))
```

### Step 2: SARSA Agent Implementation

**Q-table initialization**:
```python
self.q_table = defaultdict(lambda: defaultdict(float))
```

**ε-greedy action selection**:
```python
def get_action(self, state, training=True):
    if training and random.random() < self.epsilon:
        return random.choice(self.env.get_valid_actions(state))
    else:
        valid_actions = self.env.get_valid_actions(state)
        q_values = [self.q_table[state][action] for action in valid_actions]
        return valid_actions[np.argmax(q_values)]
```

**SARSA update**:
```python
def update_q_table(self, state, action, reward, next_state, next_action):
    current_q = self.q_table[state][action]
    next_q = self.q_table[next_state][next_action]
    
    td_target = reward + self.gamma * next_q
    td_error = td_target - current_q
    self.q_table[state][action] = current_q + self.alpha * td_error
```

### Step 3: Training Loop

**Episode structure**:
```python
def train(self, num_episodes=500):
    for episode in range(num_episodes):
        state = self.env.reset()
        action = self.get_action(state)
        
        while True:
            next_state, reward, done = self.env.step(action)
            
            if done:
                # Terminal update
                self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
                break
            
            next_action = self.get_action(next_state)
            self.update_q_table(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
```

---

## Code Walkthrough

### 1. Environment Class (`WindyGridworld`)

**Purpose**: Simulates the gridworld environment with wind effects.

**Key Methods**:
- `reset()`: Returns agent to start position
- `step(action)`: Executes action and returns (next_state, reward, done)
- `is_valid_state()`: Boundary checking
- `get_valid_actions()`: Returns available actions

**Wind Implementation**:
The wind effect is applied after the intended action:
```python
# Apply intended action
new_row = row + d_row
new_col = col + d_col

# Apply wind effect (upward wind pushes agent up)
wind_strength = self.wind[col]
new_row -= wind_strength  # Negative because upward reduces row index
```

### 2. SARSA Agent Class (`SARSAAgent`)

**Purpose**: Implements the SARSA learning algorithm.

**Key Components**:
- **Q-table**: Stores learned action values
- **Policy**: ε-greedy exploration strategy
- **Learning**: SARSA update rule implementation

**Hyperparameters**:
- **α = 0.5**: Learning rate (how much to update Q-values)
- **γ = 1.0**: Discount factor (importance of future rewards)
- **ε = 0.1**: Exploration rate (10% random actions)

### 3. Training Process

**Episode Flow**:
1. Reset environment to start state
2. Choose initial action using ε-greedy policy
3. **Repeat until goal reached**:
   - Execute action, observe reward and next state
   - Choose next action using ε-greedy policy
   - Update Q-table using SARSA rule
   - Move to next state-action pair
4. Handle terminal state update
5. Record episode statistics

### 4. Visualization

**Four-panel visualization**:
1. **Episode Lengths**: Shows learning progress over time
2. **Moving Average**: Smoothed learning curve
3. **Environment**: Grid with wind strengths marked
4. **Learned Policy**: Optimal actions for each state

---

## Results and Analysis

### Expected Learning Behavior

**Phase 1: Exploration (Episodes 1-100)**
- High variability in episode lengths
- Agent explores different paths
- Q-values are being initialized and refined

**Phase 2: Learning (Episodes 100-300)**
- Gradual decrease in episode lengths
- Agent discovers better paths
- Q-values converge toward optimal values

**Phase 3: Convergence (Episodes 300-500)**
- Stable, near-optimal episode lengths
- Consistent policy execution
- Fine-tuning of Q-values

### Optimal Path Analysis

**Without Wind**: Direct path would be 7 steps (3 right, 4 right = 7 total)

**With Wind**: Agent must compensate for upward wind:
- In columns 3-5 (wind=1): Agent may need to move down to counteract wind
- In columns 6-7 (wind=2): Strong upward wind requires careful navigation
- Optimal path typically takes 15-17 steps

### Performance Metrics

**Convergence Indicators**:
- Episode length stabilizes around 15-17 steps
- Moving average shows clear downward trend
- Policy becomes deterministic (low variance in actions)

---

## Key Insights

### 1. Wind Effect on Learning

**Challenge**: Wind creates stochastic-like behavior even in deterministic environment
**Solution**: SARSA learns to anticipate wind effects through experience

**Example**: When agent intends to move right in column 6 (wind=2):
- Intended: (3,6) → (3,7)
- Actual: (3,6) → (1,7) due to upward wind
- Agent learns to compensate by moving down first

### 2. On-Policy vs Off-Policy

**SARSA (On-Policy)**:
- Learns about the policy it's following (ε-greedy)
- More conservative, safer learning
- Considers exploration actions in value estimates

**Alternative (Q-Learning - Off-Policy)**:
- Learns about optimal policy regardless of behavior policy
- More aggressive, potentially faster convergence
- Ignores exploration in value updates

### 3. Hyperparameter Effects

**Learning Rate (α)**:
- High α: Fast learning, but unstable
- Low α: Stable learning, but slow convergence
- α = 0.5: Good balance for this problem

**Exploration Rate (ε)**:
- High ε: More exploration, slower convergence
- Low ε: Less exploration, risk of suboptimal policy
- ε = 0.1: Sufficient exploration for this environment

**Discount Factor (γ)**:
- γ = 1.0: Appropriate for episodic tasks
- Lower γ would prioritize immediate rewards

### 4. Practical Considerations

**Memory Efficiency**: 
- Q-table size: 70 states × 4 actions = 280 entries
- Manageable for this problem size

**Computational Complexity**:
- O(episodes × steps_per_episode × constant)
- Scales linearly with problem size

**Robustness**:
- Algorithm handles stochastic transitions well
- Converges reliably with proper hyperparameters

---

## Mathematical Proof Sketch: SARSA Convergence

**Theorem**: Under appropriate conditions, SARSA converges to the optimal action-value function Q*.

**Proof Outline**:

1. **Stochastic Approximation**: SARSA is a stochastic approximation algorithm of the form:
   ```
   Q_{n+1}(s,a) = Q_n(s,a) + α_n[R + γQ_n(s',a') - Q_n(s,a)]
   ```

2. **Robbins-Monro Conditions**: For convergence, we need:
   ```
   Σ_n α_n = ∞  and  Σ_n α_n² < ∞
   ```

3. **Contraction Mapping**: The expected SARSA operator T^π is a contraction:
   ```
   ||T^π Q - T^π Q'||_∞ ≤ γ||Q - Q'||_∞
   ```

4. **Policy Improvement**: ε-greedy policy improvement ensures eventual optimality as ε → 0.

**Conclusion**: SARSA converges to Q^π for the policy being followed, and with appropriate exploration, this converges to Q*.

---

## Extensions and Variations

### 1. King's Moves Windy Gridworld
Add diagonal actions (8 total actions instead of 4).

### 2. Stochastic Wind
Make wind strength probabilistic rather than deterministic.

### 3. Function Approximation
Replace Q-table with neural network for larger state spaces.

### 4. Multi-Agent Version
Multiple agents learning simultaneously in the same environment.

---

## Conclusion

The Windy Gridworld problem demonstrates key concepts in reinforcement learning:

1. **Temporal Difference Learning**: Learning from experience without a model
2. **Exploration vs Exploitation**: Balancing learning and performance
3. **Policy Evaluation**: Estimating value functions for given policies
4. **Environmental Challenges**: Handling unpredictable elements (wind)

The SARSA algorithm successfully learns an optimal policy that accounts for wind effects, showcasing the power of model-free reinforcement learning in complex environments.

This implementation provides a solid foundation for understanding more advanced RL algorithms and can be extended to tackle more complex gridworld variants and real-world navigation problems.
