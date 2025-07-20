"""
NOTE:
This implementation includes all real-world modifications of Jack’s Car Rental problem:
- Free move from location 1 to 2 (employee)
- Parking cost for more than 10 cars
- Full state space of (21 x 21)
- Action space: -5 to +5

Due to high computational complexity, the program is not run. This is for theoretical understanding only.
"""


import numpy as np
from scipy.stats import poisson
from tqdm import tqdm
import logging

logger = logging.getLogger()
logging.basicConfig(filename='carrental.log', level=logging.DEBUG)

number_of_cars = 21
actions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
gamma = 0.9
_lambda = 10

V = np.zeros((number_of_cars, number_of_cars))

logger.info("The Car Rental Problem")


logger.info(f'number of cars is {number_of_cars}')
logger.info(f'actions are {actions}')
logger.info(f'gamma is {gamma}')
logger.info(f'lambda is {_lambda}')

def policy_evaluation():
    # i and j are the number of cars at location 1 and 2
    """
    Evaluates the current policy by calculating the expected return for each state
    in the car rental problem. Iterates over all possible states (i, j), representing
    the number of cars at two locations, and computes the expected return for each
    possible action within the defined action space. The expected return is influenced
    by rental and return probabilities, rewards from rentals, movement costs between
    locations, and parking fees for excessive cars. Updates the value function V with
    the expected return for each state.
    """

    for i in range(number_of_cars):
        for j in range(number_of_cars):
            logger.debug(f'number of cars at location 1 is {i} and location 2 is {j}')
            for action in actions:
                if(i-action >= 0 and j+action <= 20):
                    logger.debug(f'Action is {action}')
                    new_i = i - action
                    new_j = j + action

                    logger.debug(f'After action, number of cars at location 1 is {new_i} and location 2 is {new_j}')

                    if action > 0:
                        move_cost = 2 * max(action - 1, 0)  # First move is free from loc 1 to loc 2
                    else:
                        move_cost = 2 * abs(action)  # All moves from loc 2 to loc 1 cost $2

                    logger.debug(f'Move cost is {move_cost}')

                    expected_return = 0

                    for rentals1 in range(11):
                        for rentals2 in range(11):
                            for return1 in range(11):
                                for return2 in range(11):
                                    # it is not equiprobable
                                    prob = (
                                        poisson.pmf(rentals1, _lambda) *
                                        poisson.pmf(rentals2, _lambda) *
                                        poisson.pmf(return1, _lambda) *
                                        poisson.pmf(return2, _lambda)
                                    )

                                    # actual number of cars returned
                                    actual_rental1 = min(rentals1, new_i)
                                    actual_rental2 = min(rentals2, new_j)

                                    # reward is +10 for each car rented and -2 for each car moved between loc1 and loc2
                                    reward = 10 * (actual_rental1 + actual_rental2) - move_cost

                                    #next state is nothing but the number of cars at location 1 and 2 after renting, returning and moving
                                    final_i = min(new_i - actual_rental1 + return1, 20)
                                    final_j = min(new_j - actual_rental2 + return2, 20)

                                    # parking fee if more than 10 cars
                                    if final_i > 10:
                                        reward -= 4

                                    if final_j > 10:
                                        reward -= 4

                                    # Expected return is the sum of all possible states
                                    expected_return += prob * (reward + gamma * V[final_i, final_j])

                    V[i, j] = expected_return
                else:
                    continue


policy = np.zeros((number_of_cars, number_of_cars))

def compute_action_value(state1, state2, action):
    """
    Computes the expected return for a given state and action in the car rental problem.

    Parameters:
    - state1 (int): Number of cars at location 1.
    - state2 (int): Number of cars at location 2.
    - action (int): Number of cars to move from location 1 to location 2.

    Returns:
    - float: The expected return for the given state and action, taking into account
      rental rewards, movement costs, and parking fees.

    The function iterates over all possible rental and return scenarios, calculating
    the probability of each scenario using the Poisson distribution. It determines
    the actual number of cars rented and computes the reward based on rentals and
    movement costs. The function also adjusts for parking fees if there are more than
    10 cars at either location after rentals and returns. The expected return is
    accumulated over all scenarios.
    """

    expected_return = 0
    for rents1 in range(11):
        for rents2 in range(11):
            for return1 in range(11):
                for return2 in range(11):
                    move_cost = 0
                    if action > 0:
                        move_cost = 2 * max(action - 1, 0)  # First move is free from loc 1 to loc 2
                    else:
                        move_cost = 2 * abs(action)
                    prob = (
                        poisson.pmf(rents1, _lambda) *
                        poisson.pmf(rents2, _lambda) *
                        poisson.pmf(return1, _lambda) *
                        poisson.pmf(return2, _lambda)
                    )

                    actual_rental1 = min(rents1, state1)
                    actual_rental2 = min(rents2, state2)

                    reward = 10 * (actual_rental1 + actual_rental2) - move_cost

                    final_i = min(state1 - actual_rental1 + return1, 20)
                    final_j = min(state2 - actual_rental2 + return2, 20)

                    if final_i > 10:
                        reward -= 4

                    if final_j > 10:
                        reward -= 4

                    expected_return += prob * (reward + gamma * V[final_i, final_j])

    return expected_return

def policy_improvement():
    """
    Improves the current policy by selecting the best action for each state
    in the car rental problem. Iterates over all possible states, represented
    by the number of cars at two locations, and evaluates all potential actions
    in the action space. For each action, the expected return is computed, and
    the action with the highest expected return is chosen as the optimal action
    for that state. The improved policy is returned, which maps each state to
    the best action based on expected returns.

    Returns:
    - numpy.ndarray: A 2D array representing the improved policy, where each
      element corresponds to the optimal action for a given state.
    """

    new_policy = np.zeros((number_of_cars, number_of_cars))

    for i in range(21):
        for j in range(21):
            best_value = float('-inf')
            best_action = 0
            for action in actions:
                if not (0 <= i - action <= 20 and 0 <= j + action <= 20):
                    continue

                expected_return = compute_action_value(i, j, action)

                if expected_return > best_value:
                    best_value = expected_return
                    best_action = action

            new_policy[i, j] = best_action

    return new_policy


for iter in tqdm(range(1000), desc='Policy Iteration'):
    policy_evaluation()
    new_policy = policy_improvement()

    if np.array_equal(policy, new_policy):
        print("Policy converged.")
        break

    policy = new_policy

print("Final Policy:")
print(policy)
