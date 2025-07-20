import numpy as np


# States for the gambler problem are 0...100 with terminal states 0 and 100
states = np.arange(101)

# Actions are stake a = [1, min(state, 100 - state)]

V = np.zeros(101)

# V(0) = 0 and V(100) = 1

V[100] = 1

probab_head = 0.55
gamma = 1


while True:
    delta = 0
    for state in states:
        stakes = np.arange(1, min(state, 100 - state) + 1)
        action_returns = []
        for stake in stakes:
            reward = 1 if state + stake == 100 else 0
            action_returns.append(probab_head * (reward + gamma * V[state + stake]) + (1 - probab_head) * (reward + gamma * V[state - stake]))
        if action_returns:  # Check if action_returns is not empty
            best_action_value = max(action_returns)
            delta = max(delta, abs(best_action_value - V[state]))
            V[state] = best_action_value
        else:
            # Handle the case when there are no valid actions for a state
            # In this case, the value of the state remains unchanged
            pass

    if delta < 1e-8:
        break

policy = np.zeros(101)

for state in states:
    print(f"V({state}) = {V[state]}")
    stakes = np.arange(1, min(state, 100 - state) + 1)
    action_returns = []
    for stake in stakes:
        reward = 1 if state + stake == 100 else 0
        action_returns.append(probab_head * (reward + gamma * V[state + stake]) + (1 - probab_head) * (reward + gamma * V[state - stake]))

    if not action_returns:
        policy[state] = 0
        continue
    policy[state] = stakes[np.argmax(action_returns)]

print("Policy:")
print(policy)

import matplotlib.pyplot as plt

# Plot Value Function
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(states, V, label='Value Function V(s)')
plt.xlabel('Capital')
plt.ylabel('Value')
plt.title('Final Value Function')
plt.grid(True)
plt.legend()

# Plot Policy
plt.subplot(1, 2, 2)
plt.plot(states, policy, label='Policy π(s)', color='orange')
plt.xlabel('Capital')
plt.ylabel('Stake')
plt.title('Final Policy (Optimal Stake at Each Capital)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



