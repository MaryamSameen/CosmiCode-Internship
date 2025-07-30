import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

# DQN Hyperparameters
env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape[0]
n_actions = env.action_space.n
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 10000
episodes = 300

# Build Q-network
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(state_shape,)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

model = build_model()
memory = deque(maxlen=memory_size)

def act(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])

def replay():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states = np.array([m[0] for m in minibatch])
    actions = np.array([m[1] for m in minibatch])
    rewards = np.array([m[2] for m in minibatch])
    next_states = np.array([m[3] for m in minibatch])
    dones = np.array([m[4] for m in minibatch])

    target_q = model.predict(states, verbose=0)
    next_q = model.predict(next_states, verbose=0)
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += gamma * np.amax(next_q[i])
        target_q[i][actions[i]] = target
    model.fit(states, target_q, epochs=1, verbose=0)

# Training loop
for ep in range(1, episodes + 1):
    state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
    total_reward = 0
    done = False
    while not done:
        action = act(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action) if len(env.step(action)) == 5 else (*env.step(action), False, {})
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        replay()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if ep % 20 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()