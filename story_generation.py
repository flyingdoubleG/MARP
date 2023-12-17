import json

from chatarena.agent import Player, Writer
from chatarena.backends import OpenAIChat
from chatarena.environments import Story
from chatarena.arena import Arena


DEFAULT_MAX_TOKENS = 4096

environment_description = "A Quarrel between two good friends about Iron Man. The story happens in two completely different settings."
controller = Player(name="Controller", backend=OpenAIChat(),
                        role_desc="You are the scene coordinator of a story. Your job is to select the next actor that should go on stage. You will be given several rounds of previous conversation in the play. If you think a player should be on stage next, print the player's name. For example: '### Next up: Amy' or '### Next up: Sheldon'. If you think the scene should end, then print '### Next up: END'.",
                        global_prompt=environment_description)
global_designer = Player(name="Global designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a story. Your job is to design a global setting for this story. The topic of your story is '{environment_description}'. Please compose a setting (sufficient but not verbose) about the background, and design the characters in this setting. For example, a valid output is:
\'The global scene is at the Yale University Bass Library.
* Jane: A 2nd year computer science undergraduate student.
* George: A first-year computer science master student.
# Jake: A professor in computer science specializing in AI.\'''', global_prompt= environment_description)
designer = Player(name="Designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a story. Previously, you have written a global setting, now your job is to design the setting of the current scene and allocate the players that should be present in this scene. Be sure to pick a player from the list given below. Your output should follow the format of <setting>\n### Next up: <character1>, <character2>, ... For example:
\'The current scene is set in a communication room.
### Next up: Brook, Elliot\'
\'The current scene is set in the grand drawing room. 
### Next up: Jane, George\'''', global_prompt= environment_description)
writer = Writer(name="Writer", backend=OpenAIChat(max_tokens=DEFAULT_MAX_TOKENS),
                 role_desc="Given the previous scene of the gameplay, your role is to convert the conversations in this gameplay into a story. Note that your output might be directly concatenated to previous outputs or future outputs, so do not structure it as a complete story, but rather a fragment of one, unless it is the first or the last scene in the story. Here are some instructions about the language: 1. Don't make the story read like a summary. 2. When crafting the story, please respect the contents of the provided conversations and in the mean time make the story coherent. 3. You can add some embellishments, but don't be verbose. 4. DON'T say you are a writer.",
                 global_prompt= environment_description)
env_manager = Player(name="Environment manager", backend=OpenAIChat(),
                     role_desc='''You are reading the script a popular play. In order to facilitate future reading, your job is to conclude each act of a player into a sentence. Only include critical information such as the actions of the player and their impacts, the emotion of the player, the change in the player's opinion or plan, etc. Your input Your output should follow the format of: ### Summary: <your summary>
Here are some examples you've previously written, follow this pattern and keep up the good work: 
Act: [Brandon]: Seated at an oak table laden with books and parchment, Brandon is the image of scholarly dedication, with a quill tucked behind one ear. The warm glow of a candle illuminates his focused expression, as he writes furiously, occasionally stopping to consult a large, leather-bound tome.
### Summary: Brandon is composing a large tome at an oak table by a candle. He is dedicated.''',
                     global_prompt= environment_description)
players = [controller, global_designer, designer, writer, env_manager]


env = Story(player_names=[p.name for p in players], max_scene_turns=20, max_scenes=2)
# arena = Arena.from_config('story_generation.json')
arena = Arena(players=players,
              environment=env, global_prompt=environment_description)
env.set_arena(arena)

# json.dump(arena.to_config(), open('story_generation.json', 'w'))
arena.run(num_steps=100)
