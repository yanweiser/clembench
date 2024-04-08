from clemgame.clemgame import GameInstanceGenerator
import numpy as np
from maps import AbstractMap
import os
import random


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'mm_mapworld'
NUM_INSTANCES = 3
GRIDS = {"small": (3,3), "medium": (3,4), "large": (4,4)}
SIZES = {"small": 4, "medium": 6, "large": 8}
SEED = 42
RANDOM_PATH = 'random_test_images'
IMAGE_PATH = os.path.join('games', 'mm_mapworld', 'resources', 'images')
MOVE_CONSTRUCTION = "GO: "
STOP_CONSTRUCTION = "DONE"
DONE_REGEX = "DONE"
MOVE_REGEX = "GO:\s*(north|east|south|west)"


def create_instances(grid_size = GRIDS['medium'], graph_size = SIZES['medium'], num_instances = NUM_INSTANCES):
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    path = os.path.join(IMAGE_PATH, RANDOM_PATH)
    imgs = np.array([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], dtype=object)
    for i in range(num_instances):
        map_images = np.random.choice(imgs, size=graph_size)
        map = AbstractMap(*grid_size, graph_size)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref = {nodes[i]: str(map_images[i]) for i in range(graph_size)}
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'start': random.choice(nodes),
            'use_images': True,
            'reprompt': False
        })
    return instances

def instance_from_args(args, prompts):
    instances = create_instances(
        grid_size=GRIDS[args['size']],
        graph_size=SIZES[args['size']],
        num_instances=NUM_INSTANCES
    )
    for i in range(len(instances)):
        if args['images']:
            instances[i]['initial_prompt'] = prompts['initial']
            instances[i]['success_response'] = prompts['later_success']
            instances[i]['invalid_response'] = prompts['later_invalid']
        else:
            instances[i]['initial_prompt'] = prompts['initial_no_img']
            instances[i]['success_response'] = prompts['later_success_no_img']
            instances[i]['invalid_response'] = prompts['later_invalid_no_img']
            instances[i]['use_images'] = False
        if args['reprompt']:
            instances[i]['reprompt'] = True
        instances[i]["reprompt_loop"] = prompts["reprompt_loop"]
        instances[i]["reprompt_format"] = prompts["reprompt_format"]
        
    return instances
        
        

class MmMapWorldInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    def on_generate(self):
        prompts = {
            'initial': self.load_template('resources/initial_prompts/prompt.template'),
            'initial_no_img': self.load_template('resources/initial_prompts/prompt_no_img.template'),
            'later_success': self.load_template('resources/later_prompts/successful_move.template'),
            'later_invalid': self.load_template('resources/later_prompts/invalid_move.template'),
            'later_success_no_img': self.load_template('resources/later_prompts/successful_move_no_img.template'),
            'later_invalid_no_img': self.load_template('resources/later_prompts/invalid_move_no_img.template'),
            'reprompt_loop': self.load_template('resources/reprompts/loop.template'),
            'reprompt_format': self.load_template('resources/reprompts/invalid_format.template'),
        }
        experiments = {
            # 'random_small': {"size": "small", "images": True, "reprompt": False},
#             'random_medium': {"size": "medium", "images": True, "reprompt": False},
            # 'random_large': {"size": "large", "images": True, "reprompt": False},
            # 'random_small_reprompt': {"size": "small", "images": True, "reprompt": True},
#             'random_medium_reprompt': {"size": "medium", "images": True, "reprompt": True},
            # 'random_large_reprompt': {"size": "large", "images": True, "reprompt": True},
            # 'random_small_no_img': {"size": "small", "images": False, "reprompt": False},
            'random_medium_no_img': {"size": "medium", "images": False, "reprompt": False},
            # 'random_large_no_img': {"size": "large", "images": False, "reprompt": False},
            # 'random_small_no_img_reprompt': {"size": "small", "images": False, "reprompt": True},
            'random_medium_no_img_reprompt': {"size": "medium", "images": False, "reprompt": True},
            # 'random_large_no_img_reprompt': {"size": "large", "images": False, "reprompt": True}
        }

        for exp in experiments.keys():
             experiment = self.add_experiment(exp)
             game_id = 0
             generated_instances = instance_from_args(experiments[exp], prompts)
             for inst in generated_instances:
                 instance = self.add_game_instance(experiment, game_id)
                 for key, value in inst.items():
                     instance[key] = value
                 instance["move_construction"] = MOVE_CONSTRUCTION
                 instance["stop_construction"] = STOP_CONSTRUCTION
                 instance["done_regex"] = DONE_REGEX
                 instance["move_regex"] = MOVE_REGEX
                 game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldInstanceGenerator().generate()

