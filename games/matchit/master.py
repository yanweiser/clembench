# was zu tun ist:
    # TODO later: interject with "do you have a question?" -> not one questioner and one answerer
    # und immer Antwort und neue Frage conkatenieren und weitergeben (also gewissermaßen 2x prompten)


# TODO: Erinnerung nach 10 turns: Komme jetzt zu einer Entscheidung

# TODO: wie mache ich das, dass im prompt von B die Beschreibung von a ist? ist das vielleicht ganz einfach?

# TODO: Score (#turns, success or not) ???

# TODO: don't give message to other player, if it's a DECISION

# TODO: log all parts of generated response (multiple flags per turn!)

# TODO: in validate: irgendwie nicht erlauben, dass mehr als eine Entscheidung getroffen wird?

# TODO: scoring mit key logging -> episode score Sachen außerhalb von turns loggen

from clemgame.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster
from clemgame import metrics as ms
from clemgame import get_logger
from clemgame.clemgame import Player
from games.matchit.instancegenerator import GAME_NAME

from typing import List, Dict

import numpy as np


logger = get_logger(__name__)

class PlayerA(Player):

    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    def _custom_response(self, messages, turn_idx) -> str:
        if turn_idx < 2:
            return "DESCRIPTION: I have a small tree in my picture, what do you have?"
        else:
            return "DECISION: Same Image."
        
class PlayerB(Player):
# is the same as PlayerA
    def __init__(self, model_name: str):
        super().__init__(model_name)
    
    def _custom_response(self, messages, turn_idx) -> str:
        if turn_idx < 2:
            return "DESCRIPTION: I have a green tree in my picture, what do you have?"
        else:
            return "DECISION: Same Image."

class MatchIt(DialogueGameMaster):

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)

        self.experiment: str = experiment["name"]
        self.flags: list= ["DESCRIPTION:", "QUESTION:", "ANSWER:", "DECISION:"]
        #self.turns = []
        self.prompt_a: str = experiment["prompt_a"]
        self.prompt_b: str = experiment["prompt_b"]

        self.solution: str = experiment["solution"] # either "Same image" or "Different Image"

        self.decision_a: bool = False
        self.decision_b: bool = False
        
        self.success_a: bool = True
        self.success_b: bool = True
        self.aborted: bool = False

        self.max_turns: int = 9

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.image_a = game_instance["image_a"]
        self.image_b = game_instance["image_b"]


        self.player_a = PlayerA(self.player_backends[0])
        self.player_b = PlayerB(self.player_backends[1])

        self.add_player(self.player_a)
        self.add_player(self.player_b)

    def _on_before_game(self):
        # add prompt to speaker message history
        self.add_user_message(self.player_a, self.prompt_a, image = self.image_a)
        self.add_user_message(self.player_b, self.prompt_b, image = self.image_b)


    def _does_game_proceed(self) -> bool:
        #if self.success == False:           # ich weiß nicht genau, ob ich das brauche.
            # self.log_message_to_self("Game over.")
            # return False
        if self.aborted:
            self.log_to_self("Game over", "Aborted")
            return False
        
        if self.decision_a and self.decision_b == True:
            return False
        if self.current_turn > self.max_turns:
            self.log_to_self("Game over", "Too many turns")
            return False

        return True
    


    def _validate_player_response(self, player: Player, utterance: str) -> bool:

        utt_parts = list(filter(None, utterance.split("\n"))) #filter to be sure that there are no empty strings
        utt_flags = []
        for part in utt_parts:
            first_word = part.split()[0]
            if first_word not in self.flags:
                self.log_to_self("invalid format", "abort")
                self.aborted = True
                return False
            else: 
                utt_flags.append(first_word.lower())
            
            if first_word == "DECISION:":
                if player == self.player_a:
                    self.decision_a = True
                    if self.solution in utterance.lower():
                        self.success_a == True
                        self.log_to_self("Decision Player A", "success")
                    else: 
                        self.success_a == False
                        self.log_to_self("Decision Player A", "loss")
                else:
                    self.decision_b = True
                    if self.solution in utterance.lower():
                        self.success_b == True
                        self.log_to_self("Decision Player B", "success")
                    else: 
                        self.success_b == False
                        self.log_to_self("Decision Player B", "loss")
        
        self.log_to_self("valid format", str(utt_flags))        
        return True



    def _after_add_player_response(self, player: Player, utterance: str):

        if player == self.player_a:
            self.add_user_message(self.player_b, utterance)
        
        if player == self.player_b:
            self.add_user_message(self.player_a, utterance)


    def compute_scores(self, episode_interactions: Dict) -> None:
        # Besides game events, the game master must also compute and log scores. 
        # Please read logdoc.md for details. In summary, there are two types of scores:
        # episode-level and turn-level. 
        #These should be computed inside the method compute_scores() 
        #using log_episode_score() and log_turn_score(), respectively.
        # compute_scores() gets interactions.json dictionary as argument, 
        #so every key and value that are necessary to compute scores should be 
        #logged into the interaction file.

        """
        - use log_episode_score to log computed episode-level scores 
        (measuring success at the whole game play) and log_turn_score 
        to log computed turn-level scores (measuring success or progress 
        at each turn)
        - all games must preferably implement the common metrics 
        (see clemgame/metrics.py and the appendix in the paper); 
        METRIC_PLAYED must not be logged, because it is inferred by the provided
         evaluation script
         - minimally, all games must compute:
             METRIC_ABORTED, 
             the binary METRIC_SUCCESS 
             and its BENCH_SCORE (which ranges form 0=fails to 100=success).
        - games can have as many additional game-specific evaluation metrics as 
        you want; make sure to use different names
        - if the game is aborted, all game-specific scores should be set to 
        ```np.nan``````
        """

        # clemscore
        # momentan erst mal: 0 wenn beide falsch, 100 wenn beide richtig und 0.5 wenn eins richtig
        # später: n_turns mit einfaktorieren

        # aborted: wenn zu viele turns ohne decision oder eine Flag nicht
        # self.log_episode_score(ms.METRIC_ABORTED, )

        # success: nur wenn beide DEcisions richtig sind, i.e. wenn 
        # "action": {
        #             "type": "Decision Player A",
        #             "content": "success"
        #         }
        all_turn_scores = []
        success_a = False
        success_b = False
        aborted = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {"request_count": 0, "violated_request_count" : 0, "parsed_request_count" : 0} 

            for event in turn:
                action = event["action"]
                # parsed requests
                if action["type"] == "invalid format":
                    turn_score_dict["violated_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["type"] == "valid format":
                    turn_score_dict["parsed_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["type"] == "Game over":
                    aborted = True
                # decision success
                elif action["type"] == "Decision Player A": # theoretically, this could occur more than once!
                    if action["content"] == "success":
                        print("success A")
                        success_a = True
                elif action["type"] == "Decision Player B": # theoretically, this could occur more than once!
                    if action["content"] == "success":
                        print("success B")
                        success_b = True
                        
            # log turn request scores   
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])

            #calculate episode scores from turn scores
            all_turn_scores.append(turn_score_dict)

            violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
            parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
            request_count = sum([turn["request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

            # log episode "success" scores
            if aborted:
                self.log_episode_score(ms.METRIC_ABORTED, 1)
                self.log_episode_score(ms.METRIC_SUCCESS, 0)
                self.log_episode_score(ms.METRIC_LOSE, 0)
                # Game-specific metrics
                self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
            else:
                # two wrong decisions:
                if not success_a and not success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 0)  # metric not applicable
                # only one decided correctly
                elif success_a != success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 50)  # current decision, may change
                    
                else:   # = success_a and success_b:    
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 1)
                    self.log_episode_score(ms.METRIC_LOSE, 0)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 100)  # metric not applicable
                print("next turn")



    

# interactions furchgehen. Die ganzen requests pro turn tracken und in dem turnscore_dict speichern
            # -> jeweils pro turn loggen. Das ist das einzige, was auf Turnebene geloggt wird
# dann am ende log_episode_score für alle turn scores in summe
# episode scores: 
            # eigener success_a, success_b
            # METRIC_SUCCESS, wenn success a & b success
            # METRIC_ABORTED, wenn irgendwo "Game over"
            # METRIC_LOSE, wenn nicht SUCCESS
            # BENCH SCORE ERST MAL einfach nur immer 100 




### Multimodal changes:
    def add_message(self, player: Player, utterance: str, role: str, image = None):
        if image is None:
            message = {"role": role, "content": utterance}
        else:
            message = {"role": role, "content": utterance, "image": image}
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str, image = None):
        self.add_message(player, utterance, role="user", image= image)
    

class MatchItBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add a description of your game
    def get_description(self):
        return "A simple game in which two players have to decide whether they see the same image or not."

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return MatchIt(experiment, player_backends)


