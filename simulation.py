import numpy as np
import time
from scoundrel_env import ScoundrelEnv 

action_dictionary = {
    0: "Enter Room 1",
    1: "Enter Room 2",
    2: "Enter Room 3",
    3: "Enter Room 4",
    4: "Skip Dungeon",
    5: "Attack with Weapon",
    6: "Attack Barehand"
}

def select_action(state, weights, env):
    q_values = np.dot(weights, state) 
    
    # === ACTION MASKING ===
    #(stopping illegal moves basically)

    # Phase B: In Combat
    if env.combat_room_index is not None:
        # Mask everything except attacking
        q_values[0] = -np.inf
        q_values[1] = -np.inf
        q_values[2] = -np.inf
        q_values[3] = -np.inf
        q_values[4] = -np.inf
        
        # Mask weapon attack if we don't have one
        if env.player.weapon is None:
            q_values[5] = -np.inf
            
    # Phase A: Exploration
    else:
        # Mask attacks
        q_values[5] = -np.inf
        q_values[6] = -np.inf
        
        # Mask skipping if it's illegal
        if env.skip_flag == 1 or env.cards_explored_this_dungeon > 0:
            q_values[4] = -np.inf
            
        # Mask entering already cleared rooms
        for i in range(4):
            if env.actions_taken_this_dungeon[i] == 1:
                q_values[i] = -np.inf
                
    # Return the highest value among the LEGAL moves
    return np.argmax(q_values)

def simulate_game():
    env = ScoundrelEnv()
    weights = np.load("trained_scoundrel_weights.npy")
    
    print("=== SCOUNDREL AI SIMULATION STARTED ===")
    time.sleep(1)
    
    state, _ = env.reset()
    starting_deck_snapshot = [str(c) for c in env.cur_dungeon.cards] + [str(c) for c in env.deck]
    done = False
    turn = 1
    
    while not done:
        print("\n" + "="*40)
        print(f"TURN {turn}")
        
        print(f"Player Health: {env.player.health} | Score: {env.player.score}")
        if env.player.weapon:
            print(f"Weapon: {env.player.weapon.value} (Max Kill: {env.player.weapon.max_defeated})")
        else:
            print("Weapon: None")
            
        print(f"Cards left in deck: {len(env.deck)}")
        print(f"Board: {env.cur_dungeon}")
        
        time.sleep(1.5) 
        
        # 3. The AI makes its choice
        action = select_action(state, weights,env)
        print(f"\n>> AI DECISION: {action_dictionary[action]} <<")
        time.sleep(1.5)
        
        # 4. Execute the step
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Catch errors if the AI did something stupid
        if reward < -5 and not terminated:
            print(">> AI made an illegal or heavily penalized move!")
        
        state = next_state
        done = terminated or truncated
        turn += 1

    # 5. Game Over Sequence
    print("\n" + "="*40)
    print("=== GAME OVER ===")
    if env.player.health <= 0:
        print("Result: AI DIED")
        print(f"Final Score: {env.player.score}")
        return False
    elif len(env.deck) < 4:
        print("Result: AI WON THE GAME!")
        print(f"Final Score: {env.player.score}")
        filename = "winning_deck.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write("=== SCOUNDREL WINNING DECK ===\n")
            
            file.write(", ".join(starting_deck_snapshot))
            
        print(f"Deck saved to {filename}!")
        return True


if __name__ == "__main__":
    simulate_game()
