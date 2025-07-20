# Reinforcement Learning Implementations

This repository contains implementations of two classic reinforcement learning problems using dynamic programming algorithms. Both problems demonstrate fundamental concepts in Markov Decision Processes (MDPs) and optimal policy computation.

## 📋 Table of Contents

- [Problems Implemented](#problems-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Algorithms](#algorithms)
- [Results](#results)
- [Performance Optimizations](#performance-optimizations)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Problems Implemented

### 1. Jack's Car Rental Problem

A classic dynamic programming problem involving optimal car allocation between two rental locations.

**Problem Description:**
- **State Space**: 21×21 grid representing number of cars at two locations (0-20 cars each)
- **Action Space**: Move -5 to +5 cars between locations
- **Objective**: Maximize expected rental revenue while minimizing car movement costs
- **Real-world Constraints**:
  - Free employee shuttle from Location 1 to Location 2 (first car moved is free)
  - Parking costs for locations with >10 cars ($4/night per excess car)
  - Poisson-distributed rental requests and returns

**Files:**
- `JacksCarRental.py` - Original implementation
- `JacksCarRental_Optimized.py` - **Highly optimized version (~120x faster)**

### 2. Gambler's Problem

A finite MDP demonstrating value iteration with a simple gambling scenario.

**Problem Description:**
- **State Space**: Capital amounts from $0 to $100
- **Action Space**: Stake amounts [1, min(capital, 100-capital)]
- **Objective**: Reach $100 with optimal betting strategy
- **Game Rules**: Biased coin flip (55% probability of heads)
- **Termination**: Game ends at $0 (lose) or $100 (win)

**File:**
- `GamblerProblem.py` - Complete implementation with visualization

## 🚀 Installation

### Prerequisites
- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- tqdm (for progress bars)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ReinforcementLearning

# Install required packages
pip install numpy scipy matplotlib tqdm

# Run the problems
python GamblerProblem.py
python JacksCarRental_Optimized.py  # Recommended optimized version
```

## 💻 Usage

### Running Jack's Car Rental Problem

```bash
# Run optimized version (recommended)
python JacksCarRental_Optimized.py

# Run original version (slower, for educational purposes)
python JacksCarRental.py
```

**Output:**
- Convergence progress with iteration details
- Final policy matrix showing optimal car movements
- Value function statistics
- Comprehensive visualizations saved as `jacks_car_rental_results.png`
- Execution time and performance metrics

### Running Gambler's Problem

```bash
python GamblerProblem.py
```

**Output:**
- Value function for each capital state
- Optimal policy (stake amounts)
- Interactive plots showing value function and policy curves

## 📁 File Structure

```
ReinforcementLearning/
├── README.md                           # This documentation
├── GamblerProblem.py                   # Gambler's problem implementation
├── JacksCarRental.py                   # Original car rental implementation
├── JacksCarRental_Optimized.py         # Optimized car rental (recommended)
├── jacks_car_rental_results.png        # Generated visualization
├── carrental.log                       # Original implementation logs
├── carrental_optimized.log             # Optimized implementation logs
└── .git/                              # Git repository files
```

## 🧮 Algorithms

### Policy Iteration (Jack's Car Rental)
1. **Policy Evaluation**: Compute value function for current policy
2. **Policy Improvement**: Update policy based on computed values
3. **Convergence Check**: Repeat until policy stabilizes

### Value Iteration (Gambler's Problem)
1. **Value Update**: Compute optimal value for each state
2. **Policy Extraction**: Derive optimal policy from value function
3. **Convergence**: Continue until value function stabilizes

## 📊 Results

### Jack's Car Rental Problem
- **Convergence**: Typically converges in 5-15 iterations
- **Optimal Strategy**: Balances rental revenue with movement costs
- **Key Insights**:
  - Free shuttle service significantly impacts optimal policy
  - Parking costs create non-linear decision boundaries
  - Higher demand locations require proactive car positioning

### Gambler's Problem
- **Optimal Policy**: Conservative betting strategy emerges
- **Value Function**: Shows probability of reaching $100 from each state
- **Key Insights**:
  - Biased coin (55% heads) creates favorable gambling conditions
  - Optimal stakes often involve "all-in" strategies near goal
  - Risk management balances potential gains with loss probability

## ⚡ Performance Optimizations

The optimized Jack's Car Rental implementation includes:

1. **Pre-computed Poisson Probabilities**: Eliminates repeated calculations
2. **Vectorized Operations**: NumPy broadcasting replaces nested loops
3. **Truncated Distributions**: Focuses computation on high-probability events
4. **State-Action Caching**: Reduces redundant computations
5. **Early Convergence**: Stops when improvement threshold is met

**Performance Improvement**: ~120x speedup over original implementation

## 📦 Dependencies

```python
numpy>=1.19.0          # Numerical computations
scipy>=1.5.0           # Statistical distributions
matplotlib>=3.3.0      # Plotting and visualization
tqdm>=4.50.0          # Progress bars
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Additional RL Problems**: Implement more classic RL scenarios
- **Algorithm Variants**: Add different solution methods (Q-learning, etc.)
- **Visualization Enhancements**: Interactive plots or animations
- **Performance Optimizations**: Further speedup techniques
- **Documentation**: Additional examples and tutorials

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Educational Value

This repository serves as an excellent resource for:
- **Students**: Learning fundamental RL concepts
- **Researchers**: Baseline implementations for comparison
- **Practitioners**: Production-ready optimized algorithms
- **Educators**: Teaching dynamic programming in RL

## 🔗 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Bellman, R. (1957). *Dynamic Programming*
- Classic RL textbook examples and problem formulations

---

**Note**: The optimized implementation (`JacksCarRental_Optimized.py`) is recommended for practical use due to its significant performance improvements while maintaining mathematical accuracy.
