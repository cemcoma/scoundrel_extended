import random
import time


# GLOBAL VARS
cards_image = {1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10",11:"J",12:"Q",13:"K",14:"A"}
suits_image = {"H":"♥️","S":"♠️","D":"♦️","C":"♣️"}
suits = ["H","S","D","C"]

#CLASSES
class card():
    def __init__(self,image,value,suit):
        self.image = image
        self.value = value
        self.suit = suit
    
    def get_value(self):
        return self.value
    
    def get_suit(self):
        return self.suit
    
    def strike_card_image(self):
        for c in self.image:
            self.image = c + '\u0336'
        self.image += suits_image[self.suit]

    def __str__(self):
        return self.image

class weapon():
    def __init__(self, value):
        self.value = value
        self.max_defeated = 15
        # self.type could be added, 
    
    def defeat_monster(self,monster_value):
        if self.max_defeated > monster_value:
            self.max_defeated = monster_value
            return max(monster_value-self.value,0)
        else:
            raise Exception("Weapon has killed equal or lower monsters, cannot be used to defeat stronger ones!")
            #print("Weapon has killed equal or lower monsters, cannot be used to defeat stronger ones!")
        
        
    def __str__(self):
        return f"Strength: {self.value}, Highest monster killed: {self.max_defeated}" 
        
class adventurer():
    def __init__(self):
        self.health = 20
        self.weapon = None
        self.score = -210

    def equip_weapon(self,weapon:weapon):
        self.weapon = weapon

    def use_potion(self,potion_val):
        self.health = min(20, self.health+potion_val)

    def attack(self,monster_val):
        self.health -= self.weapon.defeat_monster(monster_val)
        self.score += monster_val
    
    def attack_barehand(self,monster_val):
        self.health -= monster_val
        self.score += monster_val
    
    def __str__(self):
        return f"Health: {self.health} \nScore: {self.score}\nWeapon: {self.weapon or "Nothing equipped"}"

    
class dungeon():
    def __init__(self, cards:list[card]):
        self.cards = cards
        self.potion_flag = 0

    def replace_cards(self, replacements:list[card], actions: list[int]):
        """Replaces dungeon cards with 3 replacements and one non-used card from the last iteration"""
        flag = 0 #if a card in place 1,2,3 is not selected
        for i in range(4):
            if actions[i]:
                self.cards[i] = replacements[i-flag]
            else:
                flag = 1
        self.potion_flag = 0
    
    def go_to_room(self,player:adventurer, selected_room): #TODO: 3 checks for monster, potion, weapon shiiii   

        card = self.cards[selected_room]

        if card.suit == "S" or card.suit == "C":
            valid = 0
            while valid == 0:
                print(f"\nEnter the actions to take \n[1] Attack with weapon\n[2] Attack barehand")
                user_in = input("-->")
                if user_in.isnumeric() and int(user_in) in [1,2]:
                    if int(user_in) == 1:
                        try:
                            player.attack(card.value)
                            valid = 1
                        except Exception as e:      
                            print(f"{str(e)}\nHint: Try equiping a new weapon or attack barehands")    
                            time.sleep(4)
                    elif int(user_in) == 2:
                        player.attack_barehand(card.value)
                        valid = 1
                else: 
                    print("Please enter a valid input")
            print("Monster defeated!")
            print(f"Health is now {player.health}!")
        elif card.suit == "D" :
            new_weapon = weapon(card.value)
            player.equip_weapon(new_weapon)
            print(f"Weapon {new_weapon} equiped!")

        else:
            if self.potion_flag:
                print("Only 1 potion can be used per dungeon, selected card is discarded.")
            else:
                player.use_potion(card.value)
                self.potion_flag = 1
                print(f"Potion is used. Health is now {player.health} {suits_image['H']}")
    
        self.cards[selected_room].strike_card_image()

    def __str__(self):
        return f"{self.cards[0].image}, {self.cards[1].image}, {self.cards[2].image}, {self.cards[3].image}"



# GLOBAL FUNCTIONS
def generate_deck() -> list[card]:
    deck = []
    for suit in suits:
        if suit == "H" or suit == "D":
            for i in range(1,11):
                deck.append(card(suits_image[suit]+" "+cards_image[i],i,suit))
        else:
            for i in range(1,15):
                deck.append(card(suits_image[suit]+" " +cards_image[i],i,suit))
    return deck

def deck_scrambler(deck:list[card]):
    n = len(deck)
    empty = [0 for _ in range(n)]
    scrambled = [0 for _ in range(n)]

    for i in range(n):
        loc = random.randint(0,n-1)
        while empty[loc] != 0:
            loc += 1
            loc = loc % n
        scrambled[loc] = deck[i]
        empty[loc] = 1
    return scrambled 

def scoundrel():
    print()
    # Initialize game deck
    deck = generate_deck()
    scrambled_deck = deck_scrambler(deck=deck)

    #Initialize game state
    player = adventurer()
    round = 1
    skip_flag = 0

    cur_dungeon = dungeon([scrambled_deck.pop() for _ in range(4)])
   
    while player.health > 0 and len(scrambled_deck) >= 4: # TODO: add winning cond.

        print(f"******\nROUND-{round}: {cur_dungeon}")
        time.sleep(2.5)
        print(f"Enter the actions to take \n[1] Interract with the dungeon\n[2] Check stats (player, score, weapon)\n[s] Skip dungeon (can't be done consecutively)")
        user_in = input("-->")

        if user_in == "s" and skip_flag == 0:
            for i in range(4):
                scrambled_deck.insert(0,cur_dungeon.cards[i])
            cur_dungeon = dungeon([scrambled_deck.pop() for _ in range(4)])
            round += 1
            skip_flag = 1
            continue

        elif user_in == "s" and skip_flag == 1:
            print("!!! A dungeon cannot be skipped consecutively !!!")
            continue

        elif user_in == "1":
            action_count = 0
            actions = [0,0,0,0]
            
            while action_count < 3:
                print(f"{cur_dungeon}\nSelect the room you want to explore![1-4] or check player stats [5]")
                user_in = input("-->")

                if user_in.isnumeric() and int(user_in) in [1,2,3,4] and not actions[int(user_in)-1]:
                        cur_dungeon.go_to_room(player, int(user_in)-1)
                        action_count +=1
                        actions[int(user_in)-1]=1
                        if player.health <= 0:
                            break
                        
                elif user_in.isnumeric() and int(user_in) == 5:
                    print(f"** Player **\n{player}")
                    time.sleep(2.5)
                else:
                    print("Please enter a valid input! \n\n")
                    

            cur_dungeon.replace_cards([scrambled_deck.pop() for _ in range(3)], actions)
            round +=1
            skip_flag = 0
        
        elif user_in == "2":
            print(f"\n** Player **\n{player}")
            print(f"\n** Game **\nCards left: {len(scrambled_deck)}")
            time.sleep(5)
        else:
            print("Please enter a valid input! \n\n")
    
    if player.health <= 0:
        print(f"You lost!")
    elif player.health > 0:
        print(f"You won!")
        for card in scrambled_deck:
            if card.suit == "C" or card.suit == "S":
                player.score +=card.value

    return player.score


# MAIN
def main():
    user_in = 1

    while True:
        print("Welcome to scoundrel! Press 1 to play a game, press any other key to exit.")
        user_in = input("-->")
        if user_in != "1":
            break
        score = scoundrel()
        print(f"Your final score is {score}")
        time.sleep(10)
        
        print("Do you want to save your score to the leaderboards? [y/n]")
        user_in = input("-->")
        if user_in != "y":
            continue
        #TODO: leaderboards, write name yada yada

main()