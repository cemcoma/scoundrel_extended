import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from scoundrel_env import ScoundrelEnv


def select_action(state, weights, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        q_values = np.dot(weights, state) 
        return np.argmax(q_values)
    
def update_weights(weights, state, action, reward, next_state, terminated, alpha, gamma):
    # 1. Calculate the current predicted Q-value for the action taken
    current_q = np.dot(weights[action], state)
    
    # 2. Calculate the maximum expected Q-value for the next state
    if terminated:
        max_next_q = 0.0 # If the game is over, future value is 0
    else:
        next_q_values = np.dot(weights, next_state)
        max_next_q = np.max(next_q_values)
        
    # 3. Calculate the TD Target and TD Error
    td_target = reward + (gamma * max_next_q)
    td_error = td_target - current_q
    
    # 4. Update the specific row of weights
    # state is an array of features, so this scales the update for each feature
    weights[action] += alpha * td_error * state
    
    return weights

###
### TESTING & MODEL GENERATION
###

action_labels = [
    "0: Room 1", "1: Room 2", "2: Room 3", "3: Room 4", 
    "4: Skip", "5: Weapon Attack", "6: Barehand Attack"
]

feature_labels = [
    # General State (0-8)
    "Health", "Wpn Str", "Wpn Max Def", "Combat Active", 
    "Combat Mon Val", "Wpn Can Kill", "Potion Avail", "Dungeon Prog", "Can Skip",
    
    # New Global Flags (9-10)
    "Potion Exhaust", "Deck Remain",
    
    # Room 1 (11-14)
    "R1 Monster", "R1 Weapon", "R1 Potion", "R1 Can Kill",
    
    # Room 2 (15-18)
    "R2 Monster", "R2 Weapon", "R2 Potion", "R2 Can Kill",
    
    # Room 3 (19-22)
    "R3 Monster", "R3 Weapon", "R3 Potion", "R3 Can Kill",
    
    # Room 4 (23-26)
    "R4 Monster", "R4 Weapon", "R4 Potion", "R4 Can Kill",
    
    # Bias (27)
    "Bias"
]
    
if __name__ ==  "__main__":

    start = time.time()
    
    env = ScoundrelEnv()
    obs, info = env.reset()
    #print("Initial State:", obs)

    learning_rate = 0.001
    epsilon = 1.0
    epsilon_decay = 0.9995 # 15k episodes to hit minimum 
    min_epsilon = 0.05  
    
    num_actions = 7
    num_features = 28
    gamma = 0.99
    weights = np.random.uniform(low=-0.01, high=0.01, size=(num_actions, num_features))

    n = 200_000
    recent_scores = deque(maxlen=10000)

    pbar = tqdm(range(n), desc="Training Agent", unit="episodes")

    for episode in pbar:
        state, _ = env.reset()
        done = False
        
        while not done: 

            action = select_action(state, weights, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            weights = update_weights(weights, state, action, reward, next_state, terminated, learning_rate, gamma)
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
        # Progress stuff, not needed but nice to see
        recent_scores.append(env.player.score)

        if episode % 10000 == 0 and episode > 0:
            moving_avg = np.mean(recent_scores)
            pbar.set_postfix({
                'Avg Score': f"{moving_avg:.2f}", 
                'Epsilon': f"{epsilon:.4f}"
            })


    end = time.time()
    length = end - start


    #print(f"Final weights:{weights}")

    m = 1000 # number of exploitation
    print(f"\n\nFinal {m} games with weights")
    max = -1
    average = 0

    for i in range(m):
        state, _ = env.reset()
        done = False
        while not done: 
            action = select_action(state, weights, 0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated   
            state = next_state
        #print(f"{i+1}- Score:{env.player.score}")  
        average += env.player.score
        if max < env.player.score:
            max = env.player.score

    average = average / m
    print(f"This models stats: \nAverage: {average}\nMax: {max}")


    df_weights = pd.DataFrame(weights, index=action_labels, columns=feature_labels)
    plt.figure(figsize=(22, 8))
    sns.heatmap(
        df_weights, 
        annot=True,        
        fmt=".1f",         
        cmap="coolwarm",   
        center=0,          
        cbar=True
    )

    plt.title("Scoundrel RL Agent - Learned Weights")
    plt.tight_layout()
    plt.savefig("scoundrel_weights.svg", format="svg")
    # plt.show() 

    print("Testing with ", n," episodes took ", length, " seconds!")

    # saving the weights to a native numpy file
    np.save("trained_scoundrel_weights.npy", weights)
