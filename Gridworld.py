import numpy as np

np.random.seed(0)

SIZE = 5
DISCOUNT = 0.9
# left, up, right, down
ACTIONS_SEMANTICS = [(0, -1), (-1, 0), (0, 1), (1, 0)]
ACTIONS = list(range(len(ACTIONS_SEMANTICS)))
STATES = [ i * SIZE + j for i in range(SIZE) for j in range(SIZE)]

EPSILON = .1
WIND_PROB = 0
WIND_DIRECTION = 0 # left

def pretty_print_policy(policy):
  print("Policy:")
  action_labels = ["left", "up", "right", "down"]
  for x in range(SIZE):
    actions = [action_labels[policy[x * SIZE + y]] for y in range(SIZE)]
    line = " | ".join([f"{action:>5}" for action in actions])
    # line = " | ".join([f"{action[0]:>2} {action[1]:>2}" for action in actions])
    print(line)


def pretty_print_value(value):
  print("Values of states:")
  for x in range(SIZE):
    line = " | ".join([f"{value[x * SIZE + y]:<5}" for y in range(SIZE)])
    print(line)

def is_terminal(state):
    x, y = state // SIZE, state % SIZE
    return (x == 0 and y == 0) or (x == SIZE - 1 and y == SIZE - 1)

def step(state, action, wind_rate = WIND_PROB):
    if is_terminal(state):
      return state, 0
    if np.random.uniform() < wind_rate:
      state, _ = step(state, 0, wind_rate=0)
    x, y = state // SIZE, state % SIZE
    next_x = x + ACTIONS_SEMANTICS[action][0]
    next_y = y + ACTIONS_SEMANTICS[action][1]
    if next_x < 0 or next_x >= SIZE or next_y < 0 or next_y >= SIZE:
      return state, -1
    else:
      return next_x * SIZE + next_y, -1

def evaluate_deterministic_policy(policy):
  value = np.zeros(len(STATES))
  while True:
    new_value = np.zeros_like(value)
    for state in STATES:
      action = policy[state]
      next_state, reward = step(state, action)
      new_value[state] = reward + DISCOUNT * value[next_state]
    if np.max(np.abs(value - new_value)) < EPSILON:
      return np.round(new_value, decimals = 2)
    value = new_value

left_policy = {state : 0 for state in STATES}
right_policy = {state : 2 for state in STATES}


def value_iteration():
  value = np.zeros(len(STATES))
  while True:
    new_value = np.zeros_like(value)
    for state in STATES:
        values = []
        for action in ACTIONS:
          next_state, reward = step(state, action)
          values.append(reward + DISCOUNT * value[next_state])
        new_value[state] = np.max(values)
    if np.max(np.abs(new_value - value)) < EPSILON:
      return np.round(new_value, decimals = 2)
    value = new_value



def policy_iteration():
  policy = right_policy
  stable = False
  while not stable:
    # print("current policy", policy)
    stable = True
    value = evaluate_deterministic_policy(policy)
    # print("value", value)
    for state in STATES:
        best_action = None
        best_val = -100
        for action in ACTIONS:
          next_state, reward = step(state, action)
          val = reward + DISCOUNT * value[next_state]
          if val > best_val:
            best_action = action
            best_val = val
        if best_action != policy[state]:
          stable = False
          policy[state] = best_action
  return value, policy


values,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               policy = policy_iteration()



if __name__ == "__main__" :
  
  print("Right policy")
  pretty_print_value(evaluate_deterministic_policy(right_policy))
  print()

  print("Left policy")
  pretty_print_value(evaluate_deterministic_policy(left_policy))
  print()
  
  print("Value Iteration")
  pretty_print_value(value_iteration())
  print()
  
  print("Policy Iteration")
  pretty_print_value(values)
  pretty_print_policy(policy)