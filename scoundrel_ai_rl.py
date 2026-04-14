import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scoundrel
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from collections import deque


class ScoundrelEnv(gym.Env):
    def __init__(self):
        super(ScoundrelEnv, self).__init__()
        
        # ACTION SPACE
        # 0, 1, 2, 3: Interact with Room 1, 2, 3, or 4
        # 4: Skip Dungeon
        # 5: Attack with Weapon
        # 6: Attack Barehand
        self.action_space = spaces.Discrete(7)
        
        # OBSERVATION SPACE: 
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(28,), 
            dtype=np.float32
        )
        
        # game objects
        self.deck = None
        self.player = None
        self.cur_dungeon = None
        self.skip_flag = 0
        self.cards_explored_this_dungeon = 0
        self.combat_room_index = None
        self.actions_taken_this_dungeon = [0, 0, 0, 0]
        self.current_step = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        """Called at the start of every game to set things back to round 1, shuffle deck etc. etc."""
        super().reset(seed=seed)
        
        self.deck = scoundrel.deck_scrambler(scoundrel.generate_deck())
        self.player = scoundrel.adventurer()
        self.cur_dungeon = scoundrel.dungeon([self.deck.pop() for _ in range(4)])
        self.skip_flag = 0
        self.cards_explored_this_dungeon = 0
        self.actions_taken_this_dungeon = [0, 0, 0, 0]
        self.combat_room_index = None
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 500
            return self._get_obs(), reward, terminated, truncated, {}

        # ==========================================
        # PHASE B: The agent is currently in combat
        # ==========================================
        if self.combat_room_index is not None:
            if action not in [5, 6]: # Penalize any non-attack action during combat
                reward -= 10 
                return self._get_obs(), reward, terminated, truncated, {}

            monster_card = self.cur_dungeon.cards[self.combat_room_index]
            
            # Action 5: Attack with weapon
            if action == 5:
                if self.player.weapon is None:
                    reward -= 1.0 
                else:
                    try:
                        health_lost = self.player.weapon.defeat_monster(monster_card.value)
                        self.player.health -= health_lost
                        self.player.score += monster_card.value
                        
                        reward += 0.5 # Small flat reward for a successful kill
                        reward -= (health_lost * 0.1) # IMMEDIATE penalty for damage!
                    except Exception:
                        reward -= 1.0 
                        return self._get_obs(), reward, terminated, truncated, {}
            
            # Action 6: Attack barehand
            elif action == 6:
                self.player.attack_barehand(monster_card.value)
                reward += 0.5 
                # immediate penalty for face-tanking damage
                reward -= (monster_card.value * 0.1)
            
            # Combat is over! Strike the card and clear the flag
            self.cur_dungeon.cards[self.combat_room_index].strike_card_image()
            self.actions_taken_this_dungeon[self.combat_room_index] = 1
            self.combat_room_index = None
            self.cards_explored_this_dungeon += 1

        # ==========================================
        # PHASE A: The agent is exploring the dungeon
        # ==========================================
        else:
            # Logic for Actions 0, 1, 2, 3 (Selecting a room)
            if action in [0, 1, 2, 3]:
                if self.actions_taken_this_dungeon[action] == 1:
                    reward -= 5 # Penalize picking an already cleared room
                    return self._get_obs(), reward, terminated, truncated, {}
                
                card = self.cur_dungeon.cards[action]
                
                # If it's a monster, trigger the combat flag and do NOTHING else yet, 1 action per step!!!!
                if card.suit in ["S", "C"]:
                    self.combat_room_index = action
                
                # If it's a weapon
                elif card.suit == "D":
                    new_weapon = scoundrel.weapon(card.value)
                    self.player.equip_weapon(new_weapon)
                    self.cur_dungeon.cards[action].strike_card_image()
                    self.actions_taken_this_dungeon[action] = 1
                    self.cards_explored_this_dungeon += 1
                    reward += 5 # Reward for getting a weapon
                
                # If it's a potion
                elif card.suit == "H":
                    if self.cur_dungeon.potion_flag == 1:
                        reward -= 0.5 
                    else:
                        old_health = self.player.health
                        self.player.use_potion(card.value)
                        self.cur_dungeon.potion_flag = 1
                        
                        # Reward the agent based on how much it ACTUALLY healed
                        health_gained = self.player.health - old_health
                        reward += (health_gained * 0.1) 
                        
                    self.cur_dungeon.cards[action].strike_card_image()
                    self.actions_taken_this_dungeon[action] = 1
                    self.cards_explored_this_dungeon += 1

            # Logic for Action 4 (Skip)
            elif action == 4:
                if self.skip_flag == 1:
                    reward -= 10 # Penalty for trying to skip consecutively
                else:
                    # Put the 4 current cards back at the bottom of the deck
                    for i in range(4):
                        self.deck.insert(0, self.cur_dungeon.cards[i])
                    
                    self.cur_dungeon = scoundrel.dungeon([self.deck.pop() for _ in range(4)])
                    self.skip_flag = 1
                    
                    self.cards_explored_this_dungeon = 0
                    self.actions_taken_this_dungeon = [0, 0, 0, 0]
            
            # Penalize attacking when no monster is present
            elif action in [5, 6]:
                reward -= 5

        # --- Check for Dungeon Clear ---
        if self.cards_explored_this_dungeon >= 3:
            self.cur_dungeon.replace_cards([self.deck.pop() for _ in range(3)], self.actions_taken_this_dungeon)
            self.cards_explored_this_dungeon = 0
            self.actions_taken_this_dungeon = [0, 0, 0, 0]
            self.skip_flag = 0

        # --- Check for Game Over ---
        if self.player.health <= 0:
            terminated = True
            reward -= 100
        elif len(self.deck) < 4 and self.player.health > 0:
            terminated = True
            reward += 100

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Extracts an 28-dimensional feature vector from the current game state."""
        obs = np.zeros(28, dtype=np.float32) #bura feature ekliceksen değiştir
        
        # 1. Normalized Health (Max health is 20)
        obs[0] = self.player.health / 20.0
        
        # 2 & 3. Weapon Stats (Max card value is 14)
        if self.player.weapon is not None:
            obs[1] = self.player.weapon.value / 14.0
            obs[2] = self.player.weapon.max_defeated / 14.0
        else:
            obs[1] = 0.0
            obs[2] = 0.0
            
        # 4, 5 & 6. Combat State and Engineered Interactions
        if self.combat_room_index is not None:
            obs[3] = 1.0 # Combat is active
            
            monster_card = self.cur_dungeon.cards[self.combat_room_index]
            obs[4] = monster_card.value / 14.0
            
            # Engineered feature: Can our weapon actually defeat this monster?
            if self.player.weapon is not None and self.player.weapon.max_defeated > monster_card.value:
                obs[5] = 1.0
            else:
                obs[5] = 0.0
        else:
            obs[3] = 0.0
            obs[4] = 0.0
            obs[5] = 0.0
            
        # 7. Potion Availability 
        potion_available = 0.0
        if self.cur_dungeon.potion_flag == 0: # Potion hasn't been used this dungeon
            for i in range(4):
                # If room hasn't been cleared and holds a Heart
                if self.actions_taken_this_dungeon[i] == 0 and self.cur_dungeon.cards[i].suit == "H":
                    potion_available = 1.0
                    break
        obs[6] = potion_available
        
        # 8. Dungeon Progress (0.0, 0.33, 0.66, or 1.0)
        obs[7] = self.cards_explored_this_dungeon / 3.0

        # 8. Can Skip Flag (1.0 if safe to skip, 0.0 if skipping is penalized)
        if self.skip_flag == 0 and self.combat_room_index is None:
            obs[8] = 1.0 
        else:
            obs[8] = 0.0

        # 9. Potion Exhausted
        obs[9] = 1.0 if self.cur_dungeon.potion_flag == 1 else 0.0
        
        # 10. Deck Remaining
        obs[10] = len(self.deck) / 48.0
        
        # Rooms (11 through 26) - 4 features per room!
        feature_idx = 11
        for i in range(4):
            if self.actions_taken_this_dungeon[i] == 1:
                obs[feature_idx:feature_idx+4] = 0.0 # Cleared room
            else:
                card = self.cur_dungeon.cards[i]
                if card.suit in ["S", "C"]:
                    obs[feature_idx] = card.value / 14.0 # Monster Val
                    obs[feature_idx+1] = 0.0             # Weapon Val
                    obs[feature_idx+2] = 0.0             # Potion Val
                    
                    # Pre-combat Kill Check
                    if self.player.weapon is not None and self.player.weapon.max_defeated > card.value:
                        obs[feature_idx+3] = 1.0 
                    else:
                        obs[feature_idx+3] = 0.0
                        
                elif card.suit == "D":
                    obs[feature_idx:feature_idx+4] = [0.0, card.value / 14.0, 0.0, 0.0]
                elif card.suit == "H":
                    obs[feature_idx:feature_idx+4] = [0.0, 0.0, card.value / 14.0, 0.0]
                    
            feature_idx += 4
            
        # 27. Bias Term
        obs[27] = 1.0
        
        return obs
    

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
### TESTING
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
    
if __name__ == "__main__":

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
    plt.figure(figsize=(28, 8))
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
