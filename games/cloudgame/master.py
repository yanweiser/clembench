
# TODO integrate images
# TODO add to _validate_player_response: do not automatically return True (important for when not mock)
# TODO add played or aborted metric to compute_scores (see prev. todo)


import random
import copy
from typing import List, Dict, Tuple
from string import ascii_lowercase as letters

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster
from clemgame import get_logger
from clemgame.clemgame import Player
#from games.cloudgame.players import Speaker
from games.cloudgame.instancegenerator import GAME_NAME


logger = get_logger(__name__)

class Speaker(Player):
    def __init__(self, model_name: str):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model_name)
        #self.model_name: str = model_name
        #self.initial_letter: str = letter

        # a list to keep the dialogue history
        # self.history: List = []

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, messages, turn_idx) -> str:
        """Return yes or no randomly."""
        k = random.randint(0, 1)   
        if k == 0:
            return "No"
        else:
            return "Yes" 
    

class Judge(Player):

    def __init__(self, name):
        super().__init__("programmatic")
        #self.name = name

    def _custom_response(self, messages, turn_idx):
        return "That seems right."



class Cloudgame(DialogueGameMaster):
    """Implement mechanisms for playing Cloudgame."""

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters here
        self.max_words = 2
        self.turns = []
        self.allowed_words = ["yes", "no"]
        self.success = True
        self.aborted: bool = False

        self.experiment = experiment['name']
        #self.model_a = player_backends[0]
        #self.model_b = player_backends[1]
       

    def _on_setup(self, **game_instance):

        """" sets the information you specify in instances.json """
        
        self.game_instance = game_instance
        self.image = game_instance["image"]
        self.prompt = game_instance["prompt"]

        self.speaker = Speaker(self.player_backends[0])
        self.judge = Judge(self.experiment) # Argument hier ist relativ arbiträr

        self.add_player(self.speaker)
        self.add_player(self.judge)


    
    def _does_game_proceed(self):
        if len(self.turns) == 0:
            return True
        return False
    
    def _on_before_game(self):
        # add prompt to speaker message history
        self.add_user_message(self.speaker, self.prompt)#, self.image)
        # self.add_user_message(self.judge, "The game starts here.")
    

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""
        # true, wenn es wolken gibt und ja oder keine und nein -> dazu muss man schauen, wie die Instanzen aussehen
        # erst mal ist alles korrekt

        # auch schauen, ob es in ja oder nein drin ist 

        if player == self.speaker:
            true_answer = self.experiment
            if answer.lower() not in self.allowed_words:
                self.success = False
            elif answer.lower() != true_answer:
                self.success = False
          
        return True

        # if player == Speaker:
        #     if answer.lower() not in self.allowed_words:
        #         return False
            
        # if player == self.judge:
        #     if answer != "That seems right.":
        #         self.success == False
        #         return True
            
        # if player == Judge:
        #     if answer != "That seems right.":
        #         return False
            

    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.speaker:
            self.add_user_message(self.judge, utterance)
        if player == self.judge:
            self.add_user_message(self.speaker, utterance)
        
    def _on_after_turn(self, turn_idx: int):

        self.log_to_self(type_ = "judgement", value = self.success)
        self.turns.append(self.success)

    def _on_after_game(self):
        self.log_to_self(type_ = "End of game", value = "Game finished.")



    # def add_message(self, player: Player, utterance: str, role: str, image : str):
    #     message = {"role": role, "content": utterance, "image": image}
    #     history = self.messages_by_names[player.descriptor]
    #     history.append(message)

    # def add_user_message(self, player: Player, utterance: str, image : str):
    #     self.add_message(player, utterance, role="user", image= image)

        


    # from hellogame
    def compute_scores(self, episode_interactions) -> None:
        # score = 0
        # if self.success:
        #     score = 1
        # self.log_episode_score('Accuracy', score)
        ####

        for t_index, turn in enumerate(episode_interactions["turns"]):
            # player_1_message = turn[1]['action']['content']
            # jetzt kommen eigentlich die Abfragen
            score = 0
            for event in turn:
                action = event["action"]
                if action["type"] == "judgement":
                    score = action["content"]

                self.log_episode_score('Accuracy', 1 if score else 0)




class CloudgameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add a description of your game
    def get_description(self):
        return "A simple game in which a player has to decide whether they see clouds or not."

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return Cloudgame(experiment, player_backends)
