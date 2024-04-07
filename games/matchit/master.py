# was zu tun ist:
# TODO: Erinnerung nach 10 turns: Komme jetzt zu einer Entscheidung

# TODO: Score (#turns, success or not) ???

# TODO: don't give message to other player, if it's a DECISION

# TODO: log all parts of generated response (multiple flags per turn!)

# TODO: in validate: irgendwie nicht erlauben, dass mehr als eine Entscheidung getroffen wird?

# TODO: scoring mit key logging -> episode score Sachen außerhalb von turns loggen

from clemgame.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster#, GameScorer
from clemgame import metrics as ms
from clemgame import get_logger
from clemgame.clemgame import Player
from games.matchit.instancegenerator import GAME_NAME
#from backends import Model

from typing import List, Dict, Tuple

import numpy as np


logger = get_logger(__name__)


class MatchItPlayer(Player):
    # def __init__(self, backend: Model):
    #     super().__init__(backend)


    def __init__(self, model_name, role):
        super().__init__(model_name)
        self.role = role

        self.description = ""
        self.question = ""
        self.answer = ""
        self.decision = ""

        self.had_success = False

    def _custom_response(self, messages, turn_idx) -> str:
        last_message = messages[-1]["content"]


        if "collaborative" in last_message:
            logger.info("Playerdescription message here")
            return f"DESCRIPTION: from Player {self.role}"
        elif "ask" in last_message:
            return f"QUESTION: from Player {self.role}"
        elif "QUESTION" in last_message:
            logger.info("Repromt happend here")
            return f"ANSER: from Player {self.role}"
        elif "decision" in last_message:
            return "DECISION: Different image."
        else: 
            return "ANSWER: How did we land here? This is the else in the mock answers."


class MatchIt(DialogueGameMaster):
    def __init__(self, experiment: Dict, player_backends: List[str]):
    #def __init__(self, experiment: Dict, player_backends: List[Model]):
        super().__init__(GAME_NAME, experiment, player_backends)

        self.experiment: str = experiment["name"]
        self.flags: dict= experiment["flags"]
        
        self.prompt_a: str = experiment["prompt_a"] # "This is Prompt A."
        self.prompt_b: str = experiment["prompt_b"] # "This is Prompt B. Input from A: $DESCRIPTION_A$"

        self.q_reprompt = experiment["q_reprompt"] # "Reprompt: Now ask a question, starting with \"QUESTION: \""

        self.d_reprompt = experiment["d_reprompt"] # "Make a decision." 

        self.solution: str = experiment["solution"]
        
        self.success_a: bool = True
        self.success_b: bool = True
        self.aborted: bool = False

        

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.image_a = game_instance["image_a"]
        self.image_b = game_instance["image_b"]

        self.decision_turn = game_instance["decision_turn"]

        self.player_a = MatchItPlayer(self.player_backends[0], "A")
        self.player_b = MatchItPlayer(self.player_backends[1], "B")

        self.add_player(self.player_a)
        self.add_player(self.player_b)

        self.n_turns = -1
        self.answer_counter = 0 # counts how many answers a player has given per turn -> for reprompting

    def _on_before_game(self):
        # add prompt to speaker message history
        self.add_user_message(self.player_a, self.prompt_a, image = self.image_a)
        logger.info("Added Prompt A")

        #self.add_user_message(self.player_b, self.prompt_b, image = self.image_b)

    def _on_before_turn(self, turn_idx: int):
        self.n_turns += 1

    def _does_game_proceed(self) -> bool:
        #if self.success == False:           # ich weiß nicht genau, ob ich das brauche.
            # self.log_message_to_self("Game over.")
            # return False
        if self.aborted:
            self.log_to_self("Game over", "Aborted")
            return False
        
        elif self.n_turns > self.decision_turn:
            self.log_to_self("Game over", "Well done, girls!")
            return False
        else: 
            return True
    
    def check_flag(self, first_word: str, flag: str):
        if first_word == flag:
            self.log_to_self("valid format", "continue")
            return True
        else: 
            self.log_to_self("invalid format", "abort")
            self.aborted = True
            return False

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        utt_parts = list(filter(None, utterance.split("\n"))) #filter to be sure that there are no empty strings
        first_word = utt_parts[0].split(" ")[0]
        logger.info("first word = " + first_word)
        # first turn
        if self.n_turns == 0:
            if self.answer_counter == 1: # should work because answer_counter gets updated after validation
                return self.check_flag(first_word, self.flags["question"])
            else:
                return self.check_flag(first_word, self.flags["description"])
        # decision turn
        elif self.n_turns == self.decision_turn:
            if player == self.player_a and self.answer_counter == 0:
                return self.check_flag(first_word, self.flags["answer"])
            else: 
                if self.check_flag(first_word, self.flags["decision"]):
                    if utterance.lower().strip(".") == (self.flag["decision"] + " " + self.solution).lower():
                        player.success = True
                        self.log_to_self(f"Decision Player {player.role}", "success")
                    else:
                        player.success = False
                        self.log_to_self(f"Decision Player {player.role}", "loss")
                    return True
                else:
                    return False
        # all other turns
        else:
            if self.answer_counter == 0:
                return self.check_flag(first_word, self.flags["answer"])
            else: 
                return self.check_flag(first_word, self.flags["question"])



        
        # neues Konzept:
            # im ersten Turn muss es mit description anfangen
            # dann muss es answer sein, wenn answer_counter = 0 (?) und q, wenn a_c = 1
            # letzter Sonderfall: im decision turn muss für a es bei 0 wieder A sein und bei 1 decision, bei b nur decision
            
            # sonst so: keine newlines? -> könnte man bei parse auch so machen, dass es automatisch alles weitere wegschmeißt
        
        # decision runde:
            # decision loggen 

        # utt_parts = list(filter(None, utterance.split("\n"))) #filter to be sure that there are no empty strings
        # utt_flags = []
        # for part in utt_parts:
        #     first_word = part.split()[0]
        #     if first_word not in self.flags:
        #         self.log_to_self("invalid format", "abort")
        #         self.aborted = True
        #         return False
        #     else: 
        #         utt_flags.append(first_word.lower())
            
        #     if first_word == "DECISION:":
        #         if player == self.player_a:
        #             self.decision_a = True
        #             if self.solution in utterance.lower():
        #                 self.success_a == True
        #                 self.log_to_self("Decision Player A", "success")
        #             else: 
        #                 self.success_a == False
        #                 self.log_to_self("Decision Player A", "loss")
        #         else:
        #             self.decision_b = True
        #             if self.solution in utterance.lower():
        #                 self.success_b == True
        #                 self.log_to_self("Decision Player B", "success")
        #             else: 
        #                 self.success_b == False
        #                 self.log_to_self("Decision Player B", "loss")
        
        # #self.log_to_self("valid format", str(utt_flags))      
        # self.log_to_self("valid format", "continue") 
        # return True

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        #print(utterance)
        if utterance.startswith("DESCRIPTION:"):
            player.description = utterance
        elif utterance.startswith("QUESTION:"):
            player.question = utterance
        elif utterance.startswith("ANSWER:"):
            player.answer = utterance
        elif utterance.startswith("DECISION:"):
            player.decision = utterance
        
        self.answer_counter += 1
        logger.info("And helpful counter is " + str(self.answer_counter))


        utterance = utterance.split("\n") # remove anything that might not belong to the flag
        return utterance[0], True

    def _after_add_player_response(self, player: Player, utterance: str):
        # if player == self.player_a:
        #     if self.n_turns == 0:
        #         utt_filled = utterance.replace("$DESCRIPTION_A$", utterance)
        #         self.add_user_message(self.player_b, utt_filled)
        #     else:
        #         self.add_user_message(self.player_b, utterance)
        
        # if player == self.player_b:
        #     if self.n_turns == 1:
        #         self.add_user_message(self.player_b, utterance + "\n" + self.q_reprompt)
            
        #     self.add_user_message(self.player_a, utterance)

        # erste Runde: 
            # wenn player A, dann utt_filled an player  
        if self.n_turns == 0:
            if player == self.player_a:
                utt_filled = self.prompt_b.replace("$DESCRIPTION_A$", utterance)
                self.add_user_message(self.player_b, utt_filled, image = self.image_b)
            elif player == self.player_b:
                if self.player_b.description != "" and self.player_b.question != "":
                    self.add_user_message(self.player_a, self.player_b.description + "\n" + self.player_b.question)
                    self.player_b.question = ""

        # normaler Zustand:
        #  wenn answer_counter == 0, dann ist nix
        # wenn == 1, dann haben wir eine Antwort, die auch gespeichert ist und dann passiert auch erst mal nix
        # wenn == 2, dann haben wir hoffentlich sowohl Antwort als auch Frage und dann adden wir self.answers an den anderen Spieler und        #player.answers = "" machen self.answers wieder leer!
        elif self.n_turns == self.decision_turn:
            if player == self.player_a:
                self.add_user_message(self.player_b, player.answer + "\n" + player.decision + "\n" + self.d_reprompt)

        else:
            other_player = self.player_a if player == self.player_b else self.player_b
            if player.answer != "" and player.question != "":
                self.log_to_self("note", "a+q -> A:" + player.answer + " ,Q:" + player.question + " ,D:" + player.decision )
                self.add_user_message(other_player, player.answer + "\n" + player.question)
                player.description = ""
                player.question = ""
                player.answer = ""
                player.decision = ""
            elif player.decision != "" and player.question != "":
                self.log_to_self("note", "a+d -> A:" + player.answer + " ,Q:" + player.question + " ,D:" + player.decision )
                self.add_user_message(other_player, player.decision + "\n" + player.question)
                logger.info("There has been a decision and it has been added")
                player.description = ""
                player.question = ""
                player.answer = ""
                player.decision = ""
            else: 
                self.add_user_message(other_player, "DESCRIPTION: both a + q / dec+q were not filled, this is a filler.")
                self.log_to_self("note", "A:" + player.answer + " ,Q:" + player.question + " ,D:" + player.decision )

        # else:
        #     if self.answer_counter == 1:
        #         logger.info("after add player resp and hc = 1")
        #     if self.answer_counter == 2:
        #         if player == self.player_a:
        #             self.add_user_message(self.player_b, player.answers)
        #         if player == self.player_b: 
        #             self.add_user_message(self.player_a, player.answers)

    def _should_reprompt(self, player: Player):
        while self._does_game_proceed():
            if self.n_turns == 0 and player == self.player_a:
                self.answer_counter = 0
                return False
            elif self.n_turns == self.decision_turn and player == self.player_b:
                return False
            if self.answer_counter > 1: 
                self.answer_counter = 0
                return False
            return True
        return False
        # if self.should_q_reprompt == 1:
        #     return True
    
    
    def _on_before_reprompt(self, player: Player):
        if self.n_turns == self.decision_turn:
            self.add_user_message(player, self.d_reprompt)
        else:
            self.add_user_message(player, self.q_reprompt)
        



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
    


# class MatchItScorer(GameScorer):
 
#      def __init__(self, experiment: Dict, game_instance: Dict):
#          super().__init__(GAME_NAME, experiment, game_instance)
#          self.target_grid = game_instance["target_grid"] # necessary info to score the episode

#      def compute_scores(self, episode_interactions: Dict) -> None:

#         """
#         - use log_episode_score to log computed episode-level scores 
#         (measuring success at the whole game play) and log_turn_score 
#         to log computed turn-level scores (measuring success or progress 
#         at each turn)
#         - all games must preferably implement the common metrics 
#         (see clemgame/metrics.py and the appendix in the paper); 
#         METRIC_PLAYED must not be logged, because it is inferred by the provided
#          evaluation script
#          - minimally, all games must compute:
#              METRIC_ABORTED, 
#              the binary METRIC_SUCCESS 
#              and its BENCH_SCORE (which ranges form 0=fails to 100=success).
#         - games can have as many additional game-specific evaluation metrics as 
#         you want; make sure to use different names
#         - if the game is aborted, all game-specific scores should be set to 
#         ```np.nan``````
#         """

#         # clemscore
#         # momentan erst mal: 0 wenn beide falsch, 100 wenn beide richtig und 0.5 wenn eins richtig
#         # später: n_turns mit einfaktorieren

#         # aborted: wenn zu viele turns ohne decision oder eine Flag nicht
#         # self.log_episode_score(ms.METRIC_ABORTED, )

#         # success: nur wenn beide DEcisions richtig sind, i.e. wenn 
#         # "action": {
#         #             "type": "Decision Player A",
#         #             "content": "success"
#         #         }
#         all_turn_scores = []
#         success_a = False
#         success_b = False
#         aborted = False
#         for turn_idx, turn in enumerate(episode_interactions["turns"]):
#             turn_score_dict = {"request_count": 0, "violated_request_count" : 0, "parsed_request_count" : 0} 

#             for event in turn:
#                 action = event["action"]
#                 # parsed requests
#                 if action["type"] == "invalid format":
#                     turn_score_dict["violated_request_count"] += 1
#                     turn_score_dict["request_count"] += 1
#                 elif action["type"] == "valid format":
#                     turn_score_dict["parsed_request_count"] += 1
#                     turn_score_dict["request_count"] += 1
#                 elif action["type"] == "Game over":
#                     aborted = True
#                 # decision success
#                 elif action["type"] == "Decision Player A": # theoretically, this could occur more than once!
#                     if action["content"] == "success":
#                         print("success A")
#                         success_a = True
#                 elif action["type"] == "Decision Player B": # theoretically, this could occur more than once!
#                     if action["content"] == "success":
#                         print("success B")
#                         success_b = True
                        
#             # log turn request scores   
#             self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
#             self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
#             self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])

#             #calculate episode scores from turn scores
#             all_turn_scores.append(turn_score_dict)

#             violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
#             self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
#             parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
#             self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
#             request_count = sum([turn["request_count"] for turn in all_turn_scores])
#             self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

#             # log episode "success" scores
#             if aborted:
#                 self.log_episode_score(ms.METRIC_ABORTED, 1)
#                 self.log_episode_score(ms.METRIC_SUCCESS, 0)
#                 self.log_episode_score(ms.METRIC_LOSE, 0)
#                 # Game-specific metrics
#                 self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
#             else:
#                 # two wrong decisions:
#                 if not success_a and not success_b:
#                     self.log_episode_score(ms.METRIC_ABORTED, 0)
#                     self.log_episode_score(ms.METRIC_SUCCESS, 0)
#                     self.log_episode_score(ms.METRIC_LOSE, 1)
#                     # Game-specific metrics
#                     self.log_episode_score(ms.BENCH_SCORE, 0)  # metric not applicable
#                 # only one decided correctly
#                 elif success_a != success_b:
#                     self.log_episode_score(ms.METRIC_ABORTED, 0)
#                     self.log_episode_score(ms.METRIC_SUCCESS, 0)
#                     self.log_episode_score(ms.METRIC_LOSE, 1)
#                     # Game-specific metrics
#                     self.log_episode_score(ms.BENCH_SCORE, 50)  # current decision, may change
                    
#                 else:   # = success_a and success_b:    
#                     self.log_episode_score(ms.METRIC_ABORTED, 0)
#                     self.log_episode_score(ms.METRIC_SUCCESS, 1)
#                     self.log_episode_score(ms.METRIC_LOSE, 0)
#                     # Game-specific metrics
#                     self.log_episode_score(ms.BENCH_SCORE, 100)  # metric not applicable
#                 print("next turn")





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
    
    #def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
    #   return MatchItScorer(experiment, game_instance)


