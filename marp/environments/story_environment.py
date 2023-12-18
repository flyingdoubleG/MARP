import random
from typing import List, Union, Tuple

from marp.backends import OpenAIChat
from marp.config import EnvironmentConfig
from marp.environments import TimeStep
from marp.environments.base import Environment
from marp.message import MessagePool, Message
from marp.agent import SIGNAL_END_OF_CONVERSATION, Player

PLAYER_TERMINAL = 'END'


class Story(Environment):
    type_name = "Story"

    def __init__(self, player_names: List[str], max_scene_turns, max_scenes, **kwargs):
        super().__init__(player_names, **kwargs)
        self.global_message_pool = MessagePool()
        self.scene_message_pool = MessagePool()
        self._current_stage = "init"
        self._next_stage = "init"
        self._current_turn = 0
        self._current_scene = 0
        self._max_scene_turns = max_scene_turns
        self._max_scenes = max_scenes
        self._scene_start = 0  # turn where current scene starts
        self._next_player_idx = 0
        self._arena = None
        self._current_act = Message('', '', -1)
        self._role_list = []

    def set_arena(self, arena):
        self._arena = arena

    def reset(self):
        self._current_stage = "init"
        self._next_stage = "init"
        self._current_turn = 0
        self.global_message_pool.reset()
        self.scene_message_pool.reset()

    def get_next_player(self) -> str:
        if self._next_stage == "init":
            return "Global designer"
        elif self._next_stage == "scene_init":
            return "Designer"
        elif self._next_stage == "pick":
            return "Controller"
        elif self._next_stage == "impact":
            return "Summarizer"
        elif self._next_stage == "end of scene":
            return "Writer"
        else:
            return self.player_names[self._next_player_idx]

    def get_observation(self, player_name=None) -> List[Message]:
        if player_name is None:
            return self.scene_message_pool.get_all_messages()
        elif player_name == 'Summarizer':
            temp_message_pool = self.global_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.scene_message_pool.get_visible_messages(player_name, turn=self._current_turn)
            temp_message_pool.append(self._current_act)
            return temp_message_pool
        else:
            return self.global_message_pool.get_visible_messages(player_name, turn=self._current_turn) + \
                self.scene_message_pool.get_visible_messages(player_name, turn=self._current_turn)

    def print(self):
        self.global_message_pool.print()
        self.scene_message_pool.print()

    # def _controller_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
    #     message = Message(agent_name="Controller", content=text, turn=self._current_turn, visible_to=visible_to)
    #     self.global_message_pool.append_message(message)

    def is_terminal(self) -> bool:
        """
        check if the conversation is over
        """
        if self._current_scene == self._max_scenes:
            print(self._current_scene, self._max_scenes)
            return True
        # If the last message is the signal, then the conversation is over
        if self.scene_message_pool.last_message is None:
            return False
        if self.scene_message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        if self.global_message_pool.last_message is None:
            return False
        if self.global_message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        return False

    def _parse_global_designer_output(self, text: str):
        # global designer output format:
        # <setting description>
        # * <Player1>: <description>
        # * <Player2>: <description>
        # ...
        player_desc = text.split('* ')[1:]
        designed_players = [desc.split(':')[0] for desc in player_desc]
        descs = [desc.split(':')[1:] for desc in player_desc]
        for name, desc in zip(designed_players, descs):
            player = Player(name=name, role_desc=desc, backend=OpenAIChat())
            self._arena.add_player(player)
            self._role_list.append(name)
            self.player_names.append(name)
        # return all player settings, which will be added to scene message pool
        return '* ' + ''.join(player_desc)

    def _parse_designer_output(self, text: str) -> Tuple[str, List[str]]:
        try:
            setting, players = text.split('### Next up: ')
            return setting, players.split(', ')
        except ValueError:
            print('WARNING designer wrong format, using all players')
            return text, self.player_names

    def _parse_picked_player(self, text: str) -> str:
            name = text.split('Next up: ')[1]
            if name == PLAYER_TERMINAL:
                return PLAYER_TERMINAL
            for player_name in self.player_names:
                if name in player_name:
                    return player_name
            print(f'WARNING using random player, all available players are {self._role_list}')
            return random.choice(self._role_list)

    def _parse_env_manager_output(self, text: str) -> str:
        try:
            summary = text.split('### Summary:')[1]
            return summary
        except IndexError:
            print(f'WARNING env manager output format error')
            return text

    def step(self, player_name: str, action: str) -> TimeStep:
        self._current_stage = self._next_stage
        terminal = False
        if self._current_stage == "init":
            player_descs = self._parse_global_designer_output(action)
            message = Message(agent_name=player_name, content=f'Players:\n {player_descs}', turn=self._current_turn)
            self.global_message_pool.append_message(message)
            self._next_stage = "scene_init"
        elif self._current_stage == "scene_init":
            setting, players = self._parse_designer_output(action)
            # add setting to scene message pool
            message = Message(agent_name=player_name, content=setting, turn=self._current_turn)
            self.scene_message_pool.reset()
            self.scene_message_pool.append_message(message)
            # add players of current scene
            message = Message(agent_name=player_name, content=f"Players in this scene: {', '.join(players)}", turn=self._current_turn)
            self.scene_message_pool.append_message(message)
            self._scene_start = self._current_turn
            self._next_stage = "pick"
        elif self._current_stage == "pick":
            next_player = self._parse_picked_player(action)
            # controller says PLAYER_TERMINAL or max_scene_turns is reached
            if next_player == PLAYER_TERMINAL or self._current_turn - self._scene_start >= self._max_scene_turns:
                self._next_stage = "end of scene"
            else:
                self._next_player_idx = self.player_names.index(next_player)
                self._next_stage = "act"
        elif self._current_stage == "act":
            # message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            # self.scene_message_pool.append_message(message)
            self._current_act = Message(agent_name=player_name, content=f'The act you summarize: [{player_name}]: {action}', turn=self._current_turn)
            self._next_stage = "impact"
        elif self._current_stage == "impact":
            self._next_stage = "pick"
            action = self._parse_env_manager_output(action)
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.scene_message_pool.append_message(message)
        elif self._current_stage == "end of scene":
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.global_message_pool.append_message(message)
            self._current_scene += 1
            self._next_stage = "scene_init"
        terminal = terminal or self.is_terminal()
        timestep = TimeStep(observation=self.get_observation(), reward=self.get_zero_rewards(), terminal=terminal)
        self._current_turn += 1  # update current_turn every step
        return timestep

    def check_action(self, action: str, player_name: str) -> bool:
        if "As an AI language model" in action:  # GPT not act as the agent
            return False
        # if player_name == "Controller":
        #     picked_player = self._parse_picked_player(action)
        #     if picked_player not in self.player_names and picked_player != PLAYER_TERMINAL:
        #         return False
        return True

    def to_config(self) -> EnvironmentConfig:
        return super().to_config()
