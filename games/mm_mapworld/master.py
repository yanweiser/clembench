import random
from typing import List, Dict, Tuple

import utils

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster
from clemgame import get_logger
from clemgame.clemgame import Player

from clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE



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
    def __init__(self, model_name: str):
        super().__init__(model_name)
        #self.model_name: str = model_name
        # a list to keep the dialogue history
        # self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        """Return a random direction."""
        random_dir = random.choice(["north", "south", "east", "west"])
        return f'GO: {random_dir}'
    

class PathDescriber(Player):
    def __init__(self, model_name, instance_data):
        super().__init__(model_name)
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.current_room = instance_data["start"]
        self.init_prompt = instance_data["prompt"]
        self.visited_nodes=[self.current_node]
        

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
            response = "The move is not valid. You are still in the the same room. "
        else:
            response = "You have made a step and entered a different room. "
        response += "Currently available directions: "
        response += ", ".join(available_directions)
        response += " What is your next instruction?"
        return response

        
class MmMapWorld(DialogueGameMaster):
    """Implement mechanisms for playing MM-MapWorld."""

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)

        self.turns = []
        self.aborted: bool = False
        self.stop: bool = False
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
        instance_data['prompt'] = game_instance["prompt"]
        self.imgs = instance_data["imgs"]
        self.nodes = instance_data["nodes"]
        self.edges = instance_data["edges"]
        self.start = instance_data["start"]
        self.current_room = instance_data["start"]
        self.init_prompt = game_instance["prompt"]
        self.visited_nodes=[self.current_node]

        self.describer = PathDescriber('mock', instance_data)
        self.walker = PathWalker(self.player_backends[0])
        self.add_player(self.walker)
        self.add_player(self.describer)

    def _on_before_game(self):
        start_directions = self.describer.get_available_directions(self.describer.start)
        prompt = self.describer.prompt.replace('$INITIAL_DIRECTIONS$', ', '.join(start_directions))
        initial_image = self.describer.imgs[self.start]
        # add initial prompt to dialogue
        self.add_user_message(self.walker, prompt, image = initial_image)
 
    def _does_game_proceed(self):
        pass
        if not self.aborted and self.current_turn < MAX_TURNS and not self.stop:
            return True
        return False

    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""
        if player == self.walker:
            # in case we abort we set the next move to None
            self.move = None
            # Check if the answer begins with 'MOVE:'
            if answer.startswith("DONE"):
                self.stop = True
                self.log_to_self("DONE", True)
                return True
            if not answer.startswith("GO:"):
                self.aborted = True
                self.log_to_self("Invalid format", "Game aborted.")
                return False
            
            without_move = answer.replace('GO:', '')
            words = without_move.strip().split()
            new_dir = words[0]
            # the following word should be one of ['north', 'east', 'south', 'west', 'stop']
            if new_dir not in ['north', 'east', 'south', 'west', 'stop']:
                self.aborted = True
                self.log_to_self("Invalid direction", "Game aborted.")
                return False
            # everything after that can be disregarded
            
            if new_dir == 'stop':
                self.stop = True
                self.log_to_self("stop", True)
                
            self.move = words[0]
            
            self.log_to_self("Valid format", "Continue")
       
        return True
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.walker:
            self.add_user_message(self.describer, utterance)
        if player == self.describer:
            self.add_user_message(self.walker, utterance, player.imgs[self.current_room])
        
    def _on_after_turn(self, turn_idx: int):
        old_room = self.current_room
        if self.move is not None:
            self.cardinal_room_change(self.move)

        self.visited_nodes.append(self.current_room)

        self.log_to_self(type_ = "move", value = f"from {str(old_room)} to {self.current_room}")
        if self.aborted:
            self.log_to_self(type_ = "aborted", value = self.aborted)




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
    
    def compute_scores(self, episode_interactions) -> None:
        
        moves = 0
        stopped = False
        
        for turn in episode_interactions["turns"]:
            aborted = False
            
            
            for event in turn:
                action = event["action"]
                if action["type"] == "aborted":
                    if action["content"]:
                        aborted = True
                if action['type'] == "move":
                    pure = action['content'].replace('from', '')
                    pure = pure.split('to')
                    if not pure[0].strip() == pure[1].strip():
                        moves += 1
                if action['type'] == "stop":
                    if action["content"]:
                        stopped = True
                        
                        
        if aborted:
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
        else:
            self.log_episode_score(METRIC_ABORTED, 0)
            
        self.log_episode_score('moves', moves)
        self.log_episode_score('stopped', int(stopped))
                




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
                           player_backends: List[str]
                           ) -> GameMaster:
        return MmMapWorld(experiment, player_backends)
