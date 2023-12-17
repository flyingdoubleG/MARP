import json

from chatarena.agent import Player, Writer
from chatarena.backends import OpenAIChat
from chatarena.environments import Story
from chatarena.arena import Arena

DEFAULT_MAX_TOKENS = 4096

environment_description = "A Quarrel between two good friends about Iron Man."
controller = Player(name="Controller", backend=OpenAIChat(),
                        role_desc="You are the scene coordinator of a story. Your job is to select the next actor that should go on stage. You will be given several rounds of previous conversation in the play. If you think a player should be on stage next, print the player's name. For example: '### Next up: Amy' or '### Next up: Sheldon'. If you think the scene should end, then print '### Next up: END'.",
                        global_prompt=environment_description)
global_designer = Player(name="Global designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a story. Your job is to design a global setting for this story. The topic of your story is '{environment_description}'. Please compose a setting (sufficient but not verbose) about the background, and design the characters in this setting. For example, a valid output is:
\'The global scene is at the Yale University Bass Library.
* Jane: A 2nd year computer science undergraduate student.
* George: A first-year computer science master student.
# Jake: A professor in computer science specializing in AI.\'''')
designer = Player(name="Designer", backend=OpenAIChat(),
                  role_desc=f'''You are the designer of a story. Previously, you have written a global setting, now your job is to design the setting of the current scene and allocate the players that should be present in this scene. Be sure to pick a player from the list given below. Your output should follow the format of <setting>\n### Next up: <character1>, <character2>, ... For example:
\'The current scene is set in a communication room.
### Next up: Brook, Elliot\'
\'The current scene is set in the grand drawing room. 
### Next up: Jane, George\'''')
# player1 = Player(name="Sheldon", backend=OpenAIChat(),
#                  role_desc="You are a writer who has published many famous books and has many fans. You are thoughtful and friendly, and you love to read and write in general. You were saved by Annie and brought to her house after a car accident. Your leg was broken in a car accident and you need to be in a wheelchair.",
#                  global_prompt=environment_description)
# player2 = Player(name="Annie", backend=OpenAIChat(),
#                  role_desc="You're a nurse with a penchant for murder and you are also an avid fan of Sheldon's books. When you learn that Sheldon has written the death of your favorite fictional character, Bitter, you imprison Sheldon in your own home and push him to write a book to keep Bitter alive.",
#                  global_prompt=environment_description)
writer = Writer(name="Writer", backend=OpenAIChat(max_tokens=DEFAULT_MAX_TOKENS),
                 role_desc="Given several rounds of prior dialogue between characters in the scene, your role is to write an engaging and readable story, by converting the conversations into a story. Note that you should try to keep as much conversation in the story as possible. Don't make the story read like a summary. When crafting the story, please respect the contents of the provided conversations and in the mean time make the story coherent. You can add some embellishments, but don't be verbose. The most important thing is to keep important conversations, make the story read like a story, while respecting the truthfulness of the provided conversations, i.e. readable and interesting. DON'T say you are a writer.",
                 global_prompt=environment_description)

players = [controller, global_designer, designer, writer]


env = Story(player_names=[p.name for p in players], max_scene_turns=10, max_scenes=1)
# arena = Arena.from_config('story_generation.json')
arena = Arena(players=players,
              environment=env, global_prompt=environment_description)
env.set_arena(arena)

# json.dump(arena.to_config(), open('story_generation.json', 'w'))
arena.run(num_steps=100)
