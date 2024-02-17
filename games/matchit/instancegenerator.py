import sys
from clemgame.clemgame import GameInstanceGenerator

GAME_NAME = "matchit"


class MatchItInstanceGenerator(GameInstanceGenerator):
    def __init__(self, game_name):
        super().__init__(game_name)
        self.game_name = game_name

    def on_generate(self): 

        prompt_a = self.load_template('resources/initial_prompts/player_a_prompt.template')
        prompt_b = self.load_template('resources/initial_prompts/player_b_prompt.template')

        experiments = {"same_image": (self.load_csv("resources/image_pairs/same_image_10_test.csv"), "same image"), "similar_image": (self.load_csv("resources/image_pairs/similar_image_10_test.csv"), "different image"), "different_image": (self.load_csv("resources/image_pairs/different_image_10_test.csv"), "different image")}

        print(experiments)

        max_turns = 10

        for exp_name in experiments.keys(): 
            experiment =  self.add_experiment(exp_name)
            game_id = 0
            experiment["prompt_a"] = prompt_a  
            experiment["prompt_b"] = prompt_b
            experiment["solution"] = experiments[exp_name][1]


            for inst in experiments[exp_name][0]:
                game_id = game_id
                instance = self.add_game_instance(experiment, game_id)
                image_a, image_b = inst[0].strip(), inst[1].strip()
                if image_a.startswith("http"):
                    instance["image_a"] = image_a
                else:
                    instance["image_a"] = "games/matchit/resources/images/" + image_a
                if image_b.startswith("http"):
                    instance["image_b"] = image_b
                else:
                    instance["image_b"] = "games/matchit/resources/images/" + image_b
                
                max_turns = max_turns

                game_id += 1




#{game_id: , image_a, image_b, solution}, max_turns, prompt_a, prompt_b
        


        # wir haben idealerweise 3 oder 4 Listen, in denen schon die Bildpaare pro Schwierigkeit festgelegt sind (mit Dateinamen). Später kann man da vielleicht auch noch draus machen, dass diese Listen hier generiert und die Bilder heruntergeladen werden, falls sie noch nicht existieren

        #Ziel: 
        # json file with key "experiments":[] followed by list of ... experiments:
            # each experiment has: name, list of instances:   
                # instances: id, pic1, pic2, 



            # Schwierigkeit: leicht
            # game instances:
                # game id: int
                # image_player_A: str (.jpg)
                # image_player_B: str (.jpg)
                # solution: str oder int oder so (same image/1, diff image/0)
            # Schwierigkeit: mittel
                # game_id: int
                # ... etc.
            # hart:


if __name__ == "__main__":
    MatchItInstanceGenerator(GAME_NAME).generate()
