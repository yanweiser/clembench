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
DATASET_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "needed_imgs")
MAPPING_PATH = os.path.join("games", "mm_mapworld", "resources", "ade_20k", "ade_cat_instances.json")
MOVE_CONSTRUCTION = "GO: "
STOP_CONSTRUCTION = "DONE"
RESPONSE_REGEX = '{"description":\s*".+",(\s|\n)*"action":\s*".+"}'
DONE_REGEX = 'DONE'
MOVE_REGEX = 'GO:\s*(north|east|west|south)'


def create_instances(grid_size = GRIDS['medium'], graph_size = SIZES['medium'], num_instances = NUM_INSTANCES):
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    for i in range(num_instances):
        map_images = np.random.choice(imgs, size=graph_size)
        map = AbstractMap(*grid_size, graph_size)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref, cat_ref = assign_images(nodes)
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,,
            'cats': cat_ref,
            'start': random.choice(nodes),
            'use_images': True,
            'reprompt': False,
            'use_loop_warning': True,
            'use_turn_limit_warning': True
        })
    return instances

def assign_images(nodes):
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    cats = mapping.keys()
    cats_inside = [cat for cat in cats if 'outdoor' not in cat]
    chosen_cats = np.random.choice(cats_inside, size=len(nodes))
    imgs = {}
    for i in range(len(nodes)):
        cat_mapping[nodes[i]] = chosen_cats[i].split("/")[1]
        node_img = np.random.choice(mapping[chosen_cats[i]])
        imgs[nodes[i]] = os.path.join(DATASET_PATH, node_img)
    return imgs, cat_mapping

def instance_from_args(args, prompts):
    instances = create_instances(
        grid_size=GRIDS[args['size']],
        graph_size=SIZES[args['size']],
        num_instances=NUM_INSTANCES
    )
    for i in range(len(instances)):
        if args.get('one_shot', 0):
            instances[i]['initial_prompt'] = prompts['initial_one_shot']
        else:
            instances[i]['initial_prompt'] = prompts['initial']
        instances[i]['success_response'] = prompts['later_success']
        instances[i]['invalid_response'] = prompts['later_invalid']
        if args.get('reprompt', 0):
            instances[i]['reprompt'] = True
        instances[i]["reprompt_format"] = prompts["reprompt_format"]
        instances[i]["limit_warning"] = prompts["limit_warning"]
        instances[i]["loop_warning"] = prompts["loop_warning"]
    return instances
        
        

class MmMapWorldInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    def on_generate(self):
        prompts = {
            'initial': self.load_template('resources/initial_prompts/prompt.template'),
            'initial_one_shot': self.load_template('resources/initial_prompts/prompt_one_shot.template'),
            'later_success': self.load_template('resources/later_prompts/successful_move.template'),
            'later_invalid': self.load_template('resources/later_prompts/invalid_move.template'),
            'reprompt_format': self.load_template('resources/reprompts/invalid_format.template'),
            'limit_warning': self.load_template('resources/later_prompts/turn_limit.template'),
            'loop_warning': self.load_template('resources/later_prompts/loop.template'),
        }
        experiments = {
            'small': {"size": "small", "reprompt": True, "one_shot": True},
            'medium': {"size": "medium", "reprompt": True, "one_shot": True},
            'large': {"size": "large", "reprompt": True, "one_shot": True}
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
                 instance["response_regex"] = RESPONSE_REGEX
                 instance["done_regex"] = DONE_REGEX
                 instance["move_regex"] = MOVE_REGEX
                 game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldInstanceGenerator().generate()

