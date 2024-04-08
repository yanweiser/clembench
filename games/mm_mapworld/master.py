import random
from typing import List, Dict, Tuple
import re
import json
from queue import Queue
from copy import deepcopy
from time import sleep
import numpy as np

import games.mm_mapworld.utils as utils

import clemgame.metrics as ms
from backends import Model, CustomResponseModel
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster, GameScorer
from clemgame import get_logger
from clemgame.clemgame import Player

from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE, \
        BENCH_SCORE


DIRS = ["north", "south", "east", "west"]
GAME_NAME = 'mm_mapworld'
MAX_TURNS = 15

CARDINAL_TO_DELTA = {
    'north': (0, 1),
    'south': (0,-1),
    'east': (1, 0),
    'west': (-1,0)
}
DELTA_TO_CARDINAL = {
    (0, 1): 'north',
    (0,-1): 'south',
    (1, 0): 'east',
    (-1,0): 'west'
}


class PathWalker(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self, messages, turn_idx) -> str:
        """Return a random direction."""
        random_dir = random.choice(DIRS)
        return f'GO: {random_dir}'
    

class PathDescriber(Player):
    def __init__(self, model, game_instance):
        super().__init__(model)
        instance_data = utils.load_instance(game_instance)
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.current_room = instance_data["start"]
        self.success_response = game_instance["success_response"]
        self.invalid_response = game_instance["invalid_response"]
        self.visited_nodes=[self.current_room]
        
    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]
    
    def get_available_directions(self, node):
        moves = self.get_available_moves(node)
        deltas = [utils.edge_to_delta(move) for move in moves]
        cardinals = [DELTA_TO_CARDINAL[delta] for delta in deltas]
        return cardinals
    
    def cardinal_room_change(self, cardinal):
        delta = CARDINAL_TO_DELTA[cardinal]
        new_room = (self.current_room[0] + delta[0], self.current_room[1] + delta[1])
        if (self.current_room, new_room) in self.edges:
            self.current_room = new_room

    def _custom_response(self, messages, turn_idx) -> str:
        last_move = messages[-1]['content']
        without_move = last_move.replace('GO:', '')
        words = without_move.strip().split()
        new_dir = words[0]
        old_room = self.current_room
        self.cardinal_room_change(new_dir)
        invalid_direction = old_room == self.current_room
        available_directions = self.get_available_directions(self.current_room)
        if invalid_direction:
            response = self.invalid_response.replace("$DIRECTIONS$", ", ".join(available_directions))
        else:
            response = self.success_response.replace("$DIRECTIONS$", ", ".join(available_directions))
        return response

        
class MmMapWorld(DialogueGameMaster):
    """Implement mechanisms for playing MM-MapWorld."""

    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)

        self.turns = []
        self.aborted: bool = False
        self.stop: bool = False
        self.need_reprompt: bool = False
        self.did_reprompt: bool = False
        self.experiment = experiment['name']
        
    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]
    
    def get_available_directions(self, node):
        moves = self.get_available_moves(node)
        deltas = [utils.edge_to_delta(move) for move in moves]
        cardinals = [DELTA_TO_CARDINAL[delta] for delta in deltas]
        return cardinals
    
    def cardinal_room_change(self, cardinal):
        delta = CARDINAL_TO_DELTA[cardinal]
        new_room = (self.current_room[0] + delta[0], self.current_room[1] + delta[1])
        if (self.current_room, new_room) in self.edges:
            self.current_room = new_room
           
    def _on_setup(self, **game_instance):
        """" sets the information you specify in instances.json """
        self.game_instance = game_instance
        instance_data = utils.load_instance(self.game_instance)
        instance_data['initial_prompt'] = game_instance["initial_prompt"]
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.current_room = instance_data["start"]
        self.init_prompt = game_instance["initial_prompt"]
        self.visited_nodes=[self.current_room]
        
        self.done_regex = re.compile(game_instance["done_regex"])
        self.move_regex = re.compile(game_instance["move_regex"])
        
        self.done_const = game_instance["stop_construction"]
        self.move_const = game_instance["move_construction"]
        
        self.use_images = game_instance["use_images"]
        
        self.do_reprompt = game_instance["reprompt"]
        self.reprompt_loop = game_instance["reprompt_loop"]
        self.reprompt_format = game_instance["reprompt_format"]

        self.describer = PathDescriber(CustomResponseModel(), game_instance)
        self.walker = PathWalker(self.player_models[0])
        self.add_player(self.walker)
        self.add_player(self.describer)

    def _on_before_game(self):
        start_directions = self.describer.get_available_directions(self.describer.start)
        prompt = self.init_prompt.replace('$INITIAL_DIRECTIONS$', ', '.join(start_directions))
        # add initial prompt to dialogue
        if self.use_images:
            initial_image = self.describer.imgs[self.start]
            self.add_user_message(self.walker, prompt, image = initial_image)
        else:
            self.add_user_message(self.walker, prompt)
 
    def _does_game_proceed(self):
        if not self.aborted and not self.stop and self.current_turn < MAX_TURNS:
            return True
        if self.current_turn >= MAX_TURNS:
            self.log_to_self(type_ = "aborted", value = self.aborted)
            self.log_to_self("turn limit reached", True)
        return False
    
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """
        Hook

        Decide if a response utterance should be modified. If not simply return the utterance.

        When a modified utterance and a true value is returned, then a 'parse' event is logged.

        :param player: that produced the response
        :param utterance: to be potentially modified
        :return: the (modified) utterance and if to log the parse action (default: True)
        """
        if player == self.walker:
            utterance = utterance.replace("\n", "").strip()
            for word in DIRS:
                utterance = utterance.replace(word.capitalize(), word)
            done_hit = re.search(self.done_regex, utterance)
            if done_hit:
                utterance = done_hit.group()
            hit = re.search(self.move_regex, utterance)
            if hit:
                utterance = hit.group()
        return utterance, True

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""
        answer = answer.replace("\n", "").strip()
        for word in DIRS:
            answer = answer.replace(word.capitalize(), word)
        if player == self.walker:
            # in case we abort we set the next move to None
            self.move = None
            # Check if the answer begins with 'MOVE:'
            done_hit = re.search(self.done_regex, answer)
            if done_hit:
                self.stop = True
                self.log_to_self("DONE", True)
                return True
            hit = re.search(self.move_regex, answer)
            if not hit:
                if self.do_reprompt:
                    if self.did_reprompt:
                        self.aborted = True
                        self.log_to_self("Invalid format", "Game aborted.")
                        return False
                    self.need_reprompt = True
                    self.log_to_self("reprompting", "invalid format")
                    return True
                self.aborted = True
                self.log_to_self("Invalid format", "Game aborted.")
                return False
            new_dir = hit.group(1)
            self.move = new_dir
            self.log_to_self("Valid format", "Continue")
       
        return True
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.walker:
            if not self.need_reprompt or self.did_reprompt:
                self.add_user_message(self.describer, utterance)
        if player == self.describer:
            if self.use_images:
                self.add_user_message(self.walker, utterance, player.imgs[self.current_room])
            else:
                self.add_user_message(self.walker, utterance)
                
    def _should_reprompt(self, player: Player):
        if player == self.walker and self.need_reprompt and not self.did_reprompt:
            return True
        return False
    
    def _on_before_reprompt(self, player: Player):
        avail = self.get_available_directions(self.current_room)
        reprompt = self.reprompt_format
        reprompt = reprompt.replace("$DIRECTIONS$", ', '.join(avail))
        if self.use_images:
            self.add_user_message(self.walker, reprompt, self.imgs[self.current_room])
        else:
            self.add_user_message(self.walker, reprompt)
        self.did_reprompt = True
        
    def _on_after_turn(self, turn_idx: int):
        if self.aborted:
            self.log_to_self(type_ = "aborted", value = self.aborted)
        elif self.stop:
            pass
        else:
            old_room = self.current_room
            if self.move is not None:
                self.cardinal_room_change(self.move)

            self.visited_nodes.append(self.current_room)

            self.log_to_self(type_ = "move", value = json.dumps({"old": old_room, "new": self.current_room}))
        self.need_reprompt = False
        self.did_reprompt = False
            

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
        
        
    ####### scoring      
        
class MM_MapWorldScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        instance_data = utils.load_instance(self.game_instance)
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start_node = instance_data["start"]
        
    def adj(self, node):
        return set([ed[1] for ed in self.edges if ed[0] == node])
    
    def visited_all(self, visited, to_visit):
        return all([n in visited for n in to_visit])
    
    def get_available_moves(self, node, visited):
        return [edge for edge in self.edges if node == edge[0] and (edge[0] in visited or edge[1] in visited)]
    
    def find_best_moves(self, current, visited):
        to_visit = [ed[1] for ed in self.edges if ed[0] in visited and ed[1] not in visited]
        start = [current]
        q = Queue()
        q.put(start)
        found = set()
        max_len = 100
        while True:
            if not q.qsize():
                break
            n = q.get()
            if len(n) > max_len:
                break
            if self.visited_all(n, to_visit):
                found.add((n[0], n[1]))
                max_len = len(n)
                continue
            if len(n) == max_len:
                continue
            avail = self.get_available_moves(n[-1], visited)
            if all([move[1] in n for move in avail]):
                for move in avail:
                    new = deepcopy(n)
                    new.append(move[1])
                    q.put(new)
            else:
                for move in avail:
                    if not move[1] in n:
                        new = deepcopy(n)
                        new.append(move[1])
                        q.put(new)
        return found
        
    def compute_scores(self, episode_interactions) -> None:
        current = self.start_node
        seen = {self.start_node}
        seen.update(self.adj(self.start_node))
        visited = {self.start_node}
        valid_moves = 0
        invalid_moves = 0
        aborted = False
        good_move = []
        
        for turn in episode_interactions["turns"]:

            for event in turn:
                action = event["action"]
                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                if action['type'] == "move":
                    cont = json.loads(action['content'])
                    old = tuple(cont["old"])
                    new = tuple(cont["new"])
                    if not old == new:
                        valid_moves += 1
                    else:
                        invalid_moves += 1
                    
                    if not self.visited_all(visited, self.nodes) and not old == new:
                        best_moves = self.find_best_moves(old, visited)
#                         print(best_moves)
                        if (old,new) in best_moves:
                            good_move.append(True)

                        else:
                            good_move.append(False)
                    else:
                        good_move.append(False)
                    current = new
                    seen.update(self.adj(current))
                    visited.add(current)
                
                        
        if aborted:
            for i, val in enumerate(good_move):
                self.log_turn_score(i, "effiencient_move", np.NaN)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, np.NaN)
            self.log_episode_score(METRIC_LOSE, np.NaN)
            self.log_episode_score('moves', np.NaN)
            self.log_episode_score('valid_moves', np.NaN)
            self.log_episode_score('invalid_moves', np.NaN)
            self.log_episode_score('visited', np.NaN)
            self.log_episode_score('seen', np.NaN)
            self.log_episode_score('effieciency', np.NaN)
            self.log_episode_score('exploration', np.NaN)
            self.log_episode_score(BENCH_SCORE, np.NaN)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            if self.visited_all(visited, self.nodes):
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
            else:
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
            self.log_episode_score('moves', valid_moves + invalid_moves)
            self.log_episode_score('valid_moves', valid_moves)
            self.log_episode_score('invalid_moves', invalid_moves)
            self.log_episode_score('visited', len(visited))
            self.log_episode_score('seen', len(seen))
            eff = 100*sum(good_move)/len(good_move)
            self.log_episode_score('effieciency', eff)
            exp = 100*len(visited)/len(self.nodes)
            self.log_episode_score('exploration', exp)
            self.log_episode_score(BENCH_SCORE, (2*exp*eff)/(eff+exp))
        
        
                

class MmMapWorldBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return False

    # add a description of your game
    def get_description(self):
        return "In this game an agend is placed on a graph and needs to navigate through it by reasoning about past steps taken."

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_models: List[Model]
                           ) -> GameMaster:
        return MmMapWorld(experiment, player_models)
    
    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MM_MapWorldScorer(experiment, game_instance)
