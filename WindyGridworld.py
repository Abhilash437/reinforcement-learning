"""
Windy Gridworld Problem Implementation
=====================================

This module implements the Windy Gridworld problem using SARSA (State-Action-Reward-State-Action)
temporal difference learning algorithm.

The Windy Gridworld is a classic reinforcement learning problem where an agent must navigate
through a grid with wind that affects movement in certain columns.

Author: Reinforcement Learning Implementation
Date: 2025-08-02
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random


class WindyGridworld:
    """
    Windy Gridworld Environment
    
    A 7x10 grid where:
    - Start position: (3, 0)
    - Goal position: (3, 7)
    - Wind strength varies by column (upward wind)
    - Agent can move in 4 directions: up, down, left, right
    """
    
    def __init__(self):
        # Grid dimensions
        self.height = 7
        self.width = 10
        
        # Start and goal positions (row, col)
        self.start = (3, 0)
        self.goal = (3, 7)
        
        # Wind strength for each column (upward wind)
        # Index represents column, value represents wind strength
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        self.action_names = ['up', 'down', 'left', 'right']
        
        # Action effects (row_change, col_change)
        self.action_effects = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to start state"""
        self.current_state = self.start
        return self.current_state
    
    def is_valid_state(self, state):
        """Check if state is within grid boundaries"""
        row, col = state
        return 0 <= row < self.height and 0 <= col < self.width
    
    def step(self, action):
        """
        Take an action and return next state, reward, and done flag
        
        Args:
            action (int): Action to take (0-3)
            
        Returns:
            tuple: (next_state, reward, done)
        """
        row, col = self.current_state
        
        # Apply action
        d_row, d_col = self.action_effects[action]
        new_row = row + d_row
        new_col = col + d_col
        
        # Apply wind effect (upward wind)
        wind_strength = self.wind[col] if 0 <= col < len(self.wind) else 0
        new_row -= wind_strength
        
        # Ensure new state is within bounds
        new_row = max(0, min(self.height - 1, new_row))
        new_col = max(0, min(self.width - 1, new_col))
        
        next_state = (new_row, new_col)
        self.current_state = next_state
        
        # Reward is -1 for each step (encouraging shorter paths)
        reward = -1
        
        # Check if goal is reached
        done = (next_state == self.goal)
        
        return next_state, reward, done
    
    def get_valid_actions(self, state):
        """Get all valid actions from a given state"""
        return self.actions.copy()


class SARSAAgent:
    """
    SARSA Agent for solving Windy Gridworld
    
    SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference
    learning algorithm that learns the action-value function Q(s,a).
    """
    
    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1):
        """
        Initialize SARSA agent
        
        Args:
            env: Environment instance
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate for epsilon-greedy policy
        """
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table with zeros
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Statistics tracking
        self.episode_lengths = []
        self.episode_rewards = []
    
    def get_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training (bool): Whether in training mode
            
        Returns:
            int: Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: choose random action
            return random.choice(self.env.get_valid_actions(state))
        else:
            # Exploitation: choose best action
            valid_actions = self.env.get_valid_actions(state)
            q_values = [self.q_table[state][action] for action in valid_actions]
            max_q = max(q_values)
            
            # Handle ties by random selection among best actions
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        Update Q-table using SARSA update rule
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            next_action: Next action
        """
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        
        # SARSA update rule
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.alpha * td_error
    
    def train(self, num_episodes=500):
        """
        Train the agent using SARSA algorithm
        
        Args:
            num_episodes (int): Number of episodes to train
        """
        print(f"Training SARSA agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            action = self.get_action(state)
            
            episode_reward = 0
            steps = 0
            
            while True:
                # Take action and observe result
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if done:
                    # Terminal state - update Q-table with zero next Q-value
                    self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
                    break
                
                # Choose next action
                next_action = self.get_action(next_state)
                
                # Update Q-table using SARSA
                self.update_q_table(state, action, reward, next_state, next_action)
                
                # Move to next state-action pair
                state = next_state
                action = next_action
            
            # Record episode statistics
            self.episode_lengths.append(steps)
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}: Average length over last 100 episodes: {avg_length:.2f}")
    
    def get_policy(self):
        """
        Extract the learned policy from Q-table
        
        Returns:
            dict: Policy mapping states to actions
        """
        policy = {}
        for row in range(self.env.height):
            for col in range(self.env.width):
                state = (row, col)
                if state != self.env.goal:
                    policy[state] = self.get_action(state, training=False)
        return policy
    
    def test_policy(self, max_steps=1000):
        """
        Test the learned policy
        
        Args:
            max_steps (int): Maximum steps to prevent infinite loops
            
        Returns:
            tuple: (path, total_reward, success)
        """
        state = self.env.reset()
        path = [state]
        total_reward = 0
        
        for step in range(max_steps):
            action = self.get_action(state, training=False)
            next_state, reward, done = self.env.step(action)
            
            path.append(next_state)
            total_reward += reward
            
            if done:
                return path, total_reward, True
            
            state = next_state
        
        return path, total_reward, False


def visualize_results(agent, save_plots=True):
    """
    Visualize training results and learned policy
    
    Args:
        agent: Trained SARSA agent
        save_plots (bool): Whether to save plots to files
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Episode lengths over time
    ax1.plot(agent.episode_lengths)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps to Goal')
    ax1.set_title('Learning Progress: Steps to Goal vs Episode')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average of episode lengths
    window_size = 50
    if len(agent.episode_lengths) >= window_size:
        moving_avg = np.convolve(agent.episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(agent.episode_lengths)), moving_avg)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Steps (50-episode window)')
        ax2.set_title('Smoothed Learning Progress')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Grid visualization with wind
    grid = np.zeros((agent.env.height, agent.env.width))
    
    # Mark start and goal
    start_row, start_col = agent.env.start
    goal_row, goal_col = agent.env.goal
    grid[start_row, start_col] = 1  # Start
    grid[goal_row, goal_col] = 2   # Goal
    
    im = ax3.imshow(grid, cmap='viridis', origin='upper')
    ax3.set_title('Windy Gridworld Environment')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    
    # Add wind strength annotations
    for col, wind_strength in enumerate(agent.env.wind):
        ax3.text(col, agent.env.height - 0.5, f'W:{wind_strength}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    # Add start/goal labels
    ax3.text(start_col, start_row, 'S', ha='center', va='center', 
            color='white', fontweight='bold', fontsize=12)
    ax3.text(goal_col, goal_row, 'G', ha='center', va='center', 
            color='white', fontweight='bold', fontsize=12)
    
    # Plot 4: Policy visualization
    policy = agent.get_policy()
    policy_grid = np.full((agent.env.height, agent.env.width), -1)
    
    for (row, col), action in policy.items():
        policy_grid[row, col] = action
    
    # Create custom colormap for actions
    colors = ['red', 'blue', 'green', 'orange', 'white']  # up, down, left, right, empty
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    im2 = ax4.imshow(policy_grid, cmap=cmap, vmin=-1, vmax=3, origin='upper')
    ax4.set_title('Learned Policy')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    
    # Add policy arrows
    arrow_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for (row, col), action in policy.items():
        if (row, col) not in [agent.env.start, agent.env.goal]:
            ax4.text(col, row, arrow_symbols[action], ha='center', va='center', 
                    color='black', fontweight='bold', fontsize=10)
    
    # Mark start and goal on policy plot
    ax4.text(start_col, start_row, 'S', ha='center', va='center', 
            color='black', fontweight='bold', fontsize=12)
    ax4.text(goal_col, goal_row, 'G', ha='center', va='center', 
            color='black', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('/Users/abhilashhathwar/Desktop/Webdev/ReinforcementLearning/windy_gridworld_results.png', 
                   dpi=300, bbox_inches='tight')
        print("Results saved to windy_gridworld_results.png")
    
    plt.show()


def main():
    """
    Main function to run the Windy Gridworld experiment
    """
    print("=" * 60)
    print("WINDY GRIDWORLD PROBLEM - SARSA IMPLEMENTATION")
    print("=" * 60)
    
    # Create environment and agent
    env = WindyGridworld()
    agent = SARSAAgent(env, alpha=0.5, gamma=1.0, epsilon=0.1)
    
    print(f"Environment: {env.height}x{env.width} grid")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Wind pattern: {env.wind}")
    print(f"Agent parameters: α={agent.alpha}, γ={agent.gamma}, ε={agent.epsilon}")
    print()
    
    # Train the agent
    agent.train(num_episodes=500)
    
    print("\nTraining completed!")
    print(f"Final average episode length (last 100 episodes): {np.mean(agent.episode_lengths[-100:]):.2f}")
    
    # Test the learned policy
    print("\nTesting learned policy...")
    path, total_reward, success = agent.test_policy()
    
    if success:
        print(f"✓ Successfully reached goal in {len(path)-1} steps")
        print(f"Total reward: {total_reward}")
        print(f"Path: {' → '.join([str(state) for state in path])}")
    else:
        print("✗ Failed to reach goal within maximum steps")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(agent)
    
    return agent, env


if __name__ == "__main__":
    agent, env = main()
