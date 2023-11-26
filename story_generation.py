import json

from chatarena.agent import Player, Controller
from chatarena.backends import OpenAIChat
from chatarena.environments import Story
from chatarena.arena import Arena

environment_description = "You are in a house in the winter countryside with no people around."
controller = Controller(name="Controller", backend=OpenAIChat(),
                        role_desc="You are the scene coordinator of a popular play. Your job is to select the next actor that should go on stage. You will be given several rounds of previous conversation in the play. If you think a player should be on stage next, print the player's name. For example: 'Next: Amy' or 'Next: Sheldon'. If you think the scene should end, then print 'Next: END'.",
                        global_prompt="The players are Annie and Sheldon. If no previous rounds are provided, generate the first player on stage.")
global_designer = Player(name="Global designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a popular play. Your job is to design a global setting for this play. The topic of your play is '{environment_description}'. Please compose a setting that elaborate more details about the background, and design the characters in this setting. For example, a valid output is:
\'Situated in the midst of a frosted winter landscape, stands a solemn, charming, two-story countryside house, blanketed calmly by the freshly-fallen snow reflectively glowing under the moonlight.
* Winters: Positioned by the frosty window, Winters is a figure of contemplation, wrapped in a heavy woolen coat the color of midnight. He gazes out, a steaming mug in hand, his breath fogging the glass momentarily, as he observes the silent world outside.
* Brandon: Seated at an oak table laden with books and parchment, Brandon is the image of scholarly dedication, with a quill tucked behind one ear. The warm glow of a candle illuminates his focused expression, as he writes furiously, occasionally stopping to consult a large, leather-bound tome.
* George: George stands at the hearth, his back to the roaring fire, which casts dancing shadows over his rugged features. Clad in a thick, tartan-patterned sweater, he warms his hands, a slight smile playing on his lips as he listens to the muffled laughter coming from the floor above, where other members of the household are gathered.\'''')
designer = Player(name="Designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a popular play. Previously, you have written a global setting, now your job is to design the setting of the current scene and allocate the players that should be present in this scene. Be sure to pick a player from the list given below. Your output should follow the format of <setting>\n### Next up: <character1>, <character2>, ... For example:
\'Set amidst the sprawling gardens of an ancient, moonlit estate, this scene unfolds within a secluded grove, where the ruins of an old stone fountain whisper stories of bygone eras. The air is perfumed with the scent of night-blooming jasmine, and the only light comes from clusters of fireflies and the soft, eerie luminescence of phosphorescent moss that clings to the ancient stones.
### Next up: Brookelynn, Elliot\'
\'The current scene is set predominantly in the grand drawing room, illuminated merely by the soft, flickering light of the sprawling hearth. Sebastian continues to weave notes into the cold winter air with the empathic tone of his guitar. The expansive Polaroid-covered walls bear silent witness, marking segments of embroidered history. 
### Next up: Winters, George\'''')
# player1 = Player(name="Sheldon", backend=OpenAIChat(),
#                  role_desc="You are a writer who has published many famous books and has many fans. You are thoughtful and friendly, and you love to read and write in general. You were saved by Annie and brought to her house after a car accident. Your leg was broken in a car accident and you need to be in a wheelchair.",
#                  global_prompt=environment_description)
# player2 = Player(name="Annie", backend=OpenAIChat(),
#                  role_desc="You're a nurse with a penchant for murder and you are also an avid fan of Sheldon's books. When you learn that Sheldon has written the death of your favorite fictional character, Bitter, you imprison Sheldon in your own home and push him to write a book to keep Bitter alive.",
#                  global_prompt=environment_description)
writer = Player(name="Writer", backend=OpenAIChat(),
                 role_desc="You're the writer of a popular play. Your job is to write the play given several rounds of previous conversation between the characters.",
                 global_prompt=environment_description)

players = [controller, global_designer, designer, writer]


env = Story(player_names=[p.name for p in players], max_scene_turns=10, max_scenes=3)
# arena = Arena.from_config('story_generation.json')
arena = Arena(players=players,
              environment=env, global_prompt=environment_description)
env.set_arena(arena)

# json.dump(arena.to_config(), open('story_generation.json', 'w'))
arena.run(num_steps=100)
