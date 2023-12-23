
# TODO: history of player2
# TODO: self.aborted?


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
        self.history: List = []

    # implement this method as you prefer, with these same arguments
    def _custom_response(self, messages, turn_idx) -> str:
        """Return yes or no randomly."""
        k = random.randint(0, 1)   
        if k == 0:
            answer = "No"
        else:
            answer = "Yes"
        return answer
    

class Judge(Player):

    def __init__(self, name):
        super().__init__("programmatic")
        #self.name = name

    def _custom_response(self):
        return "That seems right."



class Cloudgame(DialogueGameMaster):
    """Implement mechanisms for playing Cloudgame."""

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters here
        self.max_words = 2
        self.allowed_words = ["yes", "no"]
        self.success = False
        self.aborted: bool = False

        self.experiment = experiment['name']
        self.model_a = player_backends[0]
        self.model_b = player_backends[1]
       
        
            
        
    def _on_setup(self, **game_instance):

        """" sets the information you specify in instances.json """
        
        self.game_instance = game_instance
        self.image = game_instance["image"]
        self.prompt = game_instance["prompt"]

        self.speaker = Speaker(self.player_backends[0])
        self.judge = Judge(self.player_backends[1])

        self.add_player(self.speaker)
        self.add_player(self.judge)


    
    def _does_game_proceed(self):

        return not self.aborted
    
    def _on_before_game(self):
        # add prompt to speaker message history
        self.add_user_message(self.speaker, self.prompt)
    

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""
        # true, wenn es wolken gibt und ja oder keine und nein -> dazu muss man schauen, wie die Instanzen aussehen
        # erst mal ist alles korrekt

        # auch schauen, ob es in ja oder nein drin ist 

        if player == Speaker:
            if answer.to_lower() not in self.allowed_words:
                return False

        return True
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == Speaker:
            self.add_user_message(self.judge, utterance)
        


    # from hellogame
    def compute_scores(self) -> None:
        score = 0
        if self.success:
            score = 1
        self.log_episode_score('Accuracy', score)



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
