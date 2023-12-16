import json

from chatarena.agent import Player, Writer
from chatarena.backends import OpenAIChat
from chatarena.environments import Story
from chatarena.arena import Arena

environment_description = "You are in a scary house in the winter countryside with no people around."
controller = Player(name="Controller", backend=OpenAIChat(),
                        role_desc="You are the scene coordinator of a popular play. Your job is to select the next actor that should go on stage. You will be given several rounds of previous conversation in the play. If you think a player should be on stage next, print the player's name. For example: 'Next up: Amy' or 'Next up: Sheldon'. If you think the scene should end, then print 'Next up: END'.",
                        global_prompt=environment_description)
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
writer = Writer(name="Writer", backend=OpenAIChat(max_tokens=4096),
                 role_desc="Given several rounds of prior dialogue between characters in the scene, your role is to faithfully construct a narrative for a popular play, seamlessly integrating the existing conversations into the storyline. DON'T violate any factual occurrences in the conversations, but you can appropriately add embellishments. DON'T say you are a writer.",
                 global_prompt=environment_description)
env_manager = Player(name="Environment manager", backend=OpenAIChat(),
                     role_desc='''You are reading the script a popular play. In order to facilitate future reading, your job is to conclude each act of a player into a sentence. Only include critical information such as the actions of the player and their impacts, the emotion of the player, the change in the player's opinion or plan, etc.
Here are some examples you've previously written, follow this pattern and keep up the good work: 
Act: [Brandon]: Seated at an oak table laden with books and parchment, Brandon is the image of scholarly dedication, with a quill tucked behind one ear. The warm glow of a candle illuminates his focused expression, as he writes furiously, occasionally stopping to consult a large, leather-bound tome.
Summary: Brandon is composing a large tome at an oak table by a candle. He is dedicated.''')
players = [controller, global_designer, designer, writer, env_manager]


env = Story(player_names=[p.name for p in players], max_scene_turns=10, max_scenes=3)
# arena = Arena.from_config('story_generation.json')
arena = Arena(players=players,
              environment=env, global_prompt=environment_description)
env.set_arena(arena)

# json.dump(arena.to_config(), open('story_generation.json', 'w'))
arena.run(num_steps=100)
