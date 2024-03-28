from clemgame.clemgame import GameInstanceGenerator
import numpy as np
from maps import AbstractMap
import os
import random


# set the name of the game in the script, as you named the directory
# this name will be used everywhere, including in the table of results
GAME_NAME = 'mm_mapworld'
NUM_INSTANCES = 10
GRAPH_SIZE = 8
GRID_SIZE = (4,4)
GRAPH_SIZE_SMALL = 4
GRAPH_SIZE_MEDIUM = 6
GRID_SIZE_SMALL = (3,3)
SEED = 42
RANDOM_PATH = 'random_test_images'
IMAGE_PATH = os.path.join('games', 'mm_mapworld', 'resources', 'images')
MOVE_CONSTRUCTION = "GO: "
STOP_CONSTRUCTION = "DONE"

def create_random_instances():
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    path = os.path.join(IMAGE_PATH, RANDOM_PATH)
    imgs = np.array([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], dtype=object)
    for i in range(NUM_INSTANCES):
        map_images = np.random.choice(imgs, size=GRAPH_SIZE)
        map = AbstractMap(*GRID_SIZE, GRAPH_SIZE)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref = {nodes[i]: str(map_images[i]) for i in range(GRAPH_SIZE)}
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'start': random.choice(nodes)
        })
    return instances

def create_small_random_instances():
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    path = os.path.join(IMAGE_PATH, RANDOM_PATH)
    imgs = np.array([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], dtype=object)
    for i in range(NUM_INSTANCES):
        map_images = np.random.choice(imgs, size=GRAPH_SIZE_SMALL)
        map = AbstractMap(*GRID_SIZE_SMALL, GRAPH_SIZE_SMALL)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref = {nodes[i]: str(map_images[i]) for i in range(GRAPH_SIZE_SMALL)}
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'start': random.choice(nodes)
        })
    return instances

def create_medium_random_instances():
    instances = []
    np.random.seed(SEED)
    random.seed(SEED)
    path = os.path.join(IMAGE_PATH, RANDOM_PATH)
    imgs = np.array([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))], dtype=object)
    for i in range(NUM_INSTANCES):
        map_images = np.random.choice(imgs, size=GRAPH_SIZE_MEDIUM)
        map = AbstractMap(*GRID_SIZE, GRAPH_SIZE_MEDIUM)
        nodes = [str(n) for n in map.G]
        edges = list(map.G.edges())
        rev_edges = [(edge[1], edge[0]) for edge in edges]
        edges.extend(rev_edges)
        img_ref = {nodes[i]: str(map_images[i]) for i in range(GRAPH_SIZE_MEDIUM)}
        instances.append({
            'nodes': nodes,
            'edges': [str(e) for e in edges],
            'imgs': img_ref,
            'start': random.choice(nodes)
        })
    return instances
        

class MmMapWorldInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        # always do this to initialise GameInstanceGenerator
        super().__init__(GAME_NAME)
    def on_generate(self):
      
        prompt = self.load_template('resources/initial_prompts/prompt.template')
        experiments = {
            'random': create_random_instances,
            'small_random': create_small_random_instances,
            'medium_random': create_medium_random_instances
        }

        for exp in experiments.keys():
             experiment = self.add_experiment(exp)
             game_id = 0
             generated_instances = experiments[exp]()
             for inst in generated_instances:
                 instance = self.add_game_instance(experiment, game_id)
                 for key, value in inst.items():
                     instance[key] = value
                 instance["prompt"] = prompt
                 instance["move_construction"] = MOVE_CONSTRUCTION
                 instance["stop_construction"] = STOP_CONSTRUCTION
                 game_id += 1

if __name__ == '__main__':
    # always call this, which will actually generate and save the JSON file
    MmMapWorldInstanceGenerator().generate()

