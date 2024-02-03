# TODO add to _validate_player_response: do not automatically return True (important for when not mock)
# TODO add played or aborted metric to compute_scores (see prev. todo)


import random
from typing import List, Dict, Tuple

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster
from clemgame import get_logger
from clemgame.clemgame import Player
#from games.cloudgame.players import Speaker
from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE, METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED,  METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_SUCCESS
from games.cloudgame.instancegenerator import GAME_NAME


logger = get_logger(__name__)

class Speaker(Player):
    def __init__(self, model_name: str):
        # always initialise the Player class with the model_name argument
        # if the player is a program and you don't want to make API calls to
        # LLMS, use model_name="programmatic"
        super().__init__(model_name)
        #self.model_name: str = model_name
        # a list to keep the dialogue history
        # self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        """Return yes or no randomly."""
        k = random.randint(0, 1)   
        if k == 0:
            return "Noii"
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


    #def _on_before_game(self):
        # add prompt to speaker message history
        #self.add_user_message(self.speaker, self.prompt, image = self.image)
        #self.add_user_message(self.judge, "The game starts here.")
 
    def _does_game_proceed(self):
        if not self.aborted and len(self.turns) <= 1:
            return True
        return False

   
    def _on_before_turn(self, turn_idx: int):
        
        if turn_idx == 0:
            self.add_user_message(self.speaker, self.prompt, image = self.image)
            self.add_user_message(self.judge, "Do you think this is correct?")
        if turn_idx == 1:
            self.add_user_message(self.speaker, "Are there any chickens in the picture?")
            self.add_user_message(self.judge, "Do you think this is correct?")

# TODO add played or aborted metric to compute_scores (see prev. todo)
        
    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""

        # there should never be a chicken in a picture
        if len(self.turns) != 0:
            true_answer = "no"
        
        if player == self.speaker:
            true_answer = self.experiment
            split_answer = answer.strip(".").split(" ")
            # only one word allowed
            if len(split_answer) != 1:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid word count", "Game aborted.")
                return False
    
            # if not answer.startswith("Answer:"):
            #     self.success = False
            #     self.aborted = True
            #     self.log_to_self("Invalid format", "Game aborted.")
            #     return False
            # only yes or no allowed
            if answer.lower().strip(".") not in self.allowed_words:
                self.success = False
                self.aborted = True
                self.log_to_self("Invalid words", "Game aborted.")
                return False
            # is anwer correct?
            elif answer.lower() != true_answer:
                self.success = False
            
            self.log_to_self("Valid format", "Continue")

        return True

    
    def _after_add_player_response(self, player: Player, utterance: str):
        # if player == self.speaker:
        #     self.add_user_message(self.judge, utterance)
        if player == self.judge:
            self.add_user_message(self.speaker, utterance)
        
    def _on_after_turn(self, turn_idx: int):

        self.log_to_self(type_ = "judgement", value = self.success)
        if self.aborted:
            self.log_to_self(type_ = "aborted", value = self.aborted)
        self.turns.append(self.success)

    # def _on_after_game(self):
    #     self.log_to_self(type_ = "End of game", value = "Game finished.")



    ########## Multimodal specific functions:

    def add_message(self, player: Player, utterance: str, role: str, image = None):
        if image is None:
            message = {"role": role, "content": utterance}
        else:
            message = {"role": role, "content": utterance, "image": image}
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str, image = None):
        self.add_message(player, utterance, role="user", image= image)

        


    def compute_scores(self, episode_interactions) -> None:
        # TODO: bencheval.py funktioniert damit noch nicht

        for t_index, turn in enumerate(episode_interactions["turns"]):
            # player_1_message = turn[1]['action']['content']
            # jetzt kommen eigentlich die Abfragen
            score = 0
            for event in turn:
                action = event["action"]

                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                        self.log_episode_score(METRIC_ABORTED, 1)
                    else:
                        aborted = False
                        self.log_episode_score(METRIC_ABORTED, 0)
    
                if action["type"] == "judgement":
                    score = action["content"]
        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(BENCH_SCORE, np.nan)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            self.log_episode_score(METRIC_SUCCESS, 1 if score else 0)
            self.log_episode_score(METRIC_LOSE, 0 if score else 1)
            self.log_episode_score(BENCH_SCORE, 100)
        
        # dummy values:
        self.log_episode_score(METRIC_REQUEST_COUNT, 1)
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, 1)
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED,1)
        self.log_episode_score(METRIC_REQUEST_SUCCESS, 1)



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
