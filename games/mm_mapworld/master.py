import random
from typing import List, Dict, Tuple

import numpy as np

import clemgame.metrics as ms
from clemgame.clemgame import GameMaster, GameBenchmark, DialogueGameMaster
from clemgame import get_logger
from clemgame.clemgame import Player

from instancegenerator import GAME_NAME

MAX_TURNS = 15


class Walker(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        #self.model_name: str = model_name
        # a list to keep the dialogue history
        # self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        """Return a random direction."""
        random_dir = random.choice(["north", "south", "east", "west"])
        return f'MOVE: {random_dir}'
        
        
class MmMapWorld(DialogueGameMaster):
    """Implement mechanisms for playing MM-MapWorld."""

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)

        self.turns = []
        self.aborted: bool = False
        self.stop: bool = False
        self.experiment = experiment['name']
        
    def load_nodes(self, nodes):
        """ transforms the nodes in the instance 
            from strings to tuples of ints"""
        loaded = []
        for node in nodes:
            without_brackets = node[1:-1]
            nums = without_brackets.split(',')
            tup = (int(nums[0].strip()), int(nums[1].strip()))
            loaded.append(tup)
        return loaded
    
    def load_edges(self, edges):
        """ transforms the edges in the instance 
            from strings to tuples of tuples of ints"""
        loaded = []
        for edge in edges:
            edge = edge.replace('(', '')
            edge = edge.replace(')', '')
            nums = edge.split(',')
            tup1 = (int(nums[0].strip()), int(nums[1].strip()))
            tup2 = (int(nums[2].strip()), int(nums[3].strip()))
            loaded.append((tup1, tup2))
        return loaded
    
    def load_imgs(self, imgs):
        """ changes the keys for images from strings to 
            tuples of ints """
        loaded = {}
        for key, value in imgs.items():
            key_tup = self.load_nodes([key])[0]
            loaded[key_tup] = value
        return loaded
    
    def load_start(self, start):
        """ changes the starting node from string to a
            tuple of ints """
        tup = self.load_nodes([start])[0]
        return tup
            
    def load_instance(self, instance):
        """The instance has been serialized using string for all the 
        tuples. This function reverts this process by transforming the 
        strings used to represent the graph as tuples of ints, so they
        can be worked with. 

        Args:
            instance (dict): the current instance
        """
        loaded_nodes = self.load_nodes(instance['nodes'])
        loaded_edges = self.load_edges(instance['edges'])
        loaded_imgs = self.load_imgs(instance['imgs'])
        loaded_start = self.load_start(instance['start'])
        
        return {
            'nodes': loaded_nodes,
            'edges': loaded_edges,
            'imgs': loaded_imgs,
            'start': loaded_start
        }
        

    def get_available_moves(self, node):
        return [edge for edge in self.edges if node == edge[0]]
    
    def edge_to_delta(self, edge):
        dx = edge[1][0] - edge[0][0]
        dy = edge[1][1] - edge[0][1]
        return (dx, dy)
    
    def delta_to_cardinal(self, delta):
        transformation = {
            (0, 1): 'north',
            (0,-1): 'south',
            (1, 0): 'east',
            (-1,0): 'west'
        }
        return transformation[delta]
    
    def cardinal_to_delta(self, cardinal):
        transformation = {
            'north': (0, 1),
            'south': (0,-1),
            'east': (1, 0),
            'west': (-1,0)
        }
        return transformation[cardinal]
    
    def get_available_directions(self, node):
        moves = self.get_available_moves(node)
        deltas = [self.edge_to_delta(move) for move in moves]
        cardinals = [self.delta_to_cardinal(delta) for delta in deltas]
        return cardinals
    
    def cardinal_room_change(self, cardinal):
        delta = self.delta_to_cardinal(cardinal)
        new_room = (self.current_room[0] + delta[0], self.current_room[1] + delta[1])
        if (self.current_room, new_room) in self.edges:
            self.current_room = new_room

    def visited_all(nodes, visited):
        return all([n in visited for n in nodes])

       

    def _on_setup(self, **game_instance):
        """" sets the information you specify in instances.json """
        
        self.game_instance = game_instance
        self.instance_data = self.load_instance(game_instance)
        self.imgs = self.instance_data["imgs"]
        self.nodes = self.instance_data["nodes"]
        self.edges = self.instance_data["edges"]
        self.start = self.instance_data["start"]
        self.current_room = self.instance_data["start"]
        self.prompt = game_instance["prompt"]

        self.walker = Walker(self.player_backends[0])
        self.add_player(self.walker)


    def _on_before_game(self):
        start_directions = self.get_available_directions(self.start)
        self.prompt = self.prompt.replace('[DIR]', ', '.join(start_directions))
        initial_image = self.imgs[self.start]
        # add initial prompt to dialogue
        self.add_user_message(self.walker, self.prompt, image = initial_image)
 
    def _does_game_proceed(self):
        pass
        if not self.aborted and self.current_turn < MAX_TURNS and not self.stop:
            return True
        return False

   
    def _on_before_turn(self, turn_idx: int):
        # in the first turn, _on_before_game already takes care of this
        if turn_idx == 0:
            return
        # after the first turn: 
        dirs = self.get_available_directions(self.current_room)
        img = self.imgs[self.current_room]
        if self.invalid_direction:
            msg = "Your room did not change since the direction you gave was not valid. " 
        else:
            msg = "You are now in this room. "
        msg += f"From here you can move in the following directions:\n{', '.join(dirs)}"
        self.add_user_message(self.walker, msg, image=img)
            
 
    def _validate_player_response(self, player: Player, answer: str) -> bool:
        """Check if the utterance conforms to rules (cloudgame specific)."""
        # in case we abort we set the next move to None
        self.move = None
    
        # Check if the answer begins with 'MOVE:'
        if not answer.startswith("MOVE:"):
            self.aborted = True
            self.log_to_self("Invalid format", "Game aborted.")
            return False
        
        without_move = answer.replace('MOVE:', '')
        words = without_move.strip().split()

        # the following word should be one of ['north', 'east', 'south', 'west', 'stop']
        if words[0] in ['north', 'east', 'south', 'west', 'stop']:
            self.aborted = True
            self.log_to_self("Invalid direction", "Game aborted.")
            return False
        # everything after that can be disregarded
        
        # TBD:
        # check if the given direction is a valid one? (Is there an edge from the current
        # room to the next one?)
        
        self.move = words[0]
        
        self.log_to_self("Valid format", "Continue")
        return True
        

    
    def _after_add_player_response(self, player: Player, utterance: str):
        pass
        
    def _on_after_turn(self, turn_idx: int):
        old_room = self.current_room
        if self.move is not None:
            self.cardinal_room_change(self.move)
        self.invalid_direction = False
        if old_room == self.current_room:
            self.invalid_direction = True

        self.log_to_self(type_ = "move", value = f"from {str(old_room)} to {self.current_room}")
        if self.aborted:
            self.log_to_self(type_ = "aborted", value = self.aborted)

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



class MmMapWorldBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    # defines whether the game is single player or not
    def is_single_player(self):
        return True

    # add a description of your game
    def get_description(self):
        return "In this game an agend is placed on a graph and needs to navigate through it by reasoning about past steps taken."

    # copy this, replacing the name of the game master in the return statement
    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return MmMapWorld(experiment, player_backends)
