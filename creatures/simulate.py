import random
import threading
import time
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
import numpy as np

from utils.logging import log


class Strategy:
    """
    Base class for a particular strategy that an agent might use to choose its
    actions.

    For example, a strategy might involve a set of rules or heuristics, or it
    might involve a machine learning model. It might have methods for updating
    the model parameters, making predictions, and visualizing the strategy.

    Subclasses must implement the `action` method, which calcs the next
    action to take based on the states of the environment, game, and agent.
    """

    def action(env: "Environment", game: "Game", agent: "Agent"):
        """
        Calculates the next action to take based on the states of the
        environment, game, and agent.

        Returns:
            The action to be taken in the current state.
        """
        pass


class RetreatStrategy(Strategy):
    def action(**_):
        return "retreat"


class AttackStrategy(Strategy):
    def action(**_):
        return "attack"


class RetaliateStrategy(Strategy):
    def action(game: "Game", **_):
        return game.state.opposer.state.last_non_skip_move or "retreat"


class ExploitStrategy(Strategy):
    def action(game: "Game", **_):
        def _opposite(last_move):
            if last_move == "attack":
                return "retreat"
            elif last_move == "retreat":
                return "attack"
            else:
                return None

        return _opposite(game.state.opposer.state.last_non_skip_move) or "retreat"


class Agent:
    """
    This class represents an agent in the simulation. It has a state that
    changes over time, based on the actions it takes and the feedback it
    receives from the environment. The Agent class might have methods for
    choosing an action, updating its state based on the feedback from the
    environment, and visualizing its own state
    """

    class State:
        def __init__(self, init_hp):
            self.hp = init_hp
            self.xp = 0
            self.room_ids = []
            self.moves: List[List[Tuple[str, bool]]] = []

        def _last_move(self, is_this_game=False, is_skip_ok=True):
            for game in reversed(self.moves):
                for move, _ in reversed(game):
                    if is_skip_ok or move != "skip":
                        return move
                if is_this_game:
                    break

        @property
        def last_move(self):
            return self._last_move()

        @property
        def last_non_skip_move(self):
            return self._last_move(is_skip_ok=False)

    def __init__(
        self,
        idx: str,
        init_hp: int,
        strategy: "Strategy",
    ):
        self.idx = idx
        self.init_hp = init_hp
        self.strategy = strategy
        self.state = self.State(init_hp)


class Game:
    """
    Represents a game that the agents are playing. It typically has rules that
    determine the payoffs for each agent, based on the actions they choose.

    It might have methods for computing the payoffs, visualizing the game
    matrix, and simulating multiple rounds of the game.

    Attributes:
        state: An instance of the State class, which represents the current
               state of the game.
    """

    class State:
        def __init__(self):
            self.moves: List[Tuple[str, bool]] = []
            self.mover: "Agent" = None
            self.opposer: "Agent" = None

        def _last_move(self, is_of_opposer=True, is_skip_ok=True):
            is_this_game = True
            if is_of_opposer:
                return self.opposer.state._last_move(is_this_game, is_skip_ok)
            else:
                return self.mover.state._last_move(is_this_game, is_skip_ok)

        @property
        def opposer_last_move(self):
            return self._last_move()

        @property
        def mover_last_move(self):
            return self._last_move(is_of_opposer=False)

        @property
        def opposer_last_non_skip_move(self):
            return self._last_move(is_skip_ok=False)

        @property
        def mover_last_non_skip_move(self):
            return self._last_move(is_of_opposer=False, is_skip_ok=False)

    def __init__(
        self,
        round_idx: str,
        room_idx: str,
        agents: Tuple["Agent"],
        prob_skip: float,
        prob_attack_success: float,
        prob_retreat_success: float,
        hp_attack_success_opposer: int,
        hp_attack_success_mover: int,
        hp_attack_fail_opposer: int,
        hp_attack_fail_mover: int,
        hp_retreat_success_opposer: int,
        hp_retreat_success_mover: int,
        hp_retreat_fail_opposer: int,
        hp_retreat_fail_mover: int,
        xp_attack_success_opposer: int,
        xp_attack_success_mover: int,
        xp_attack_fail_opposer: int,
        xp_attack_fail_mover: int,
        xp_retreat_success_opposer: int,
        xp_retreat_success_mover: int,
        xp_retreat_fail_opposer: int,
        xp_retreat_fail_mover: int,
        hp_bonus: int,
        xp_bonus: int,
        is_hp_reset: bool,
        is_xp_reset: bool,
        **_,
    ):
        self.round_idx = round_idx
        self.room_idx = room_idx
        for agent in agents:
            agent.state.room_ids.append(room_idx)
        self.agents = agents
        self.prob_skip = prob_skip
        self.prob_attack_success = prob_attack_success
        self.prob_retreat_success = prob_retreat_success
        self.hp_attack_success_opposer = hp_attack_success_opposer
        self.hp_attack_success_mover = hp_attack_success_mover
        self.hp_attack_fail_opposer = hp_attack_fail_opposer
        self.hp_attack_fail_mover = hp_attack_fail_mover
        self.hp_retreat_success_opposer = hp_retreat_success_opposer
        self.hp_retreat_success_mover = hp_retreat_success_mover
        self.hp_retreat_fail_opposer = hp_retreat_fail_opposer
        self.hp_retreat_fail_mover = hp_retreat_fail_mover
        self.xp_attack_success_opposer = xp_attack_success_opposer
        self.xp_attack_success_mover = xp_attack_success_mover
        self.xp_attack_fail_opposer = xp_attack_fail_opposer
        self.xp_attack_fail_mover = xp_attack_fail_mover
        self.xp_retreat_success_opposer = xp_retreat_success_opposer
        self.xp_retreat_success_mover = xp_retreat_success_mover
        self.xp_retreat_fail_opposer = xp_retreat_fail_opposer
        self.xp_retreat_fail_mover = xp_retreat_fail_mover
        self.hp_bonus = hp_bonus
        self.xp_bonus = xp_bonus
        self.is_hp_reset = is_hp_reset
        self.is_xp_reset = is_xp_reset
        self.state = self.State()

    def run(self, env):
        for agent in self.agents:
            if agent.state.hp >= 0:
                agent.state.hp += self.hp_bonus
                agent.state.xp += self.xp_bonus
            if self.is_hp_reset:
                agent.state.hp = agent.init_hp
            if self.is_xp_reset:
                agent.state.xp = 0
            agent.state.moves.append([])
        agent_a = self.agents[0]
        agent_b = self.agents[1]
        game_title = (
            f"Round {str(self.round_idx).zfill(4)} - "
            f"Room {str(self.room_idx).zfill(4)} - "
            f"{str(agent_a.idx).zfill(4)}({agent_a.strategy}) vs "
            f"{str(agent_b.idx).zfill(4)}({agent_b.strategy})"
        )
        log.debug(f"BEGIN: {game_title}")
        while not self._game_ended():
            if self.state.mover == agent_a:
                self.state.mover = agent_b
                self.state.opposer = agent_a
            else:
                self.state.mover = agent_a
                self.state.opposer = agent_b
            self._process_turn(env)
        log.debug(f"END:   {game_title}")

    def _game_ended(self):
        if not self.state.mover or not self.state.opposer:
            return False
        if self.state.mover.state.hp < 0 or self.state.opposer.state.hp < 0:
            return True
        if self.state.moves[-1][0] == "retreat" and self.state.moves[-1][1]:
            return True
        return False

    def _process_turn(self, env):
        if random.random() < self.prob_skip:
            self._skip()
        else:
            action = self.state.mover.strategy.action(
                env=env, game=self, agent=self.state.mover
            )
            if action == "attack":
                self._attack()
            elif action == "retreat":
                self._retreat()
            else:
                raise ValueError(f"Invalid Action, {action}")

    def _skip(self):
        move = ("skip", None)
        self.state.moves.append(move)
        self.state.mover.state.moves[-1].append(move)

    def _attack(self):
        if random.random() < self.prob_attack_success:
            self.state.opposer.state.hp += self.hp_attack_success_opposer
            self.state.mover.state.hp += self.hp_attack_success_mover
            self.state.opposer.state.xp += self.xp_attack_success_opposer
            self.state.mover.state.xp += self.xp_attack_success_mover
            move = ("attack", True)
            self.state.moves.append(move)
            self.state.mover.state.moves[-1].append(move)
        else:
            self.state.opposer.state.hp += self.hp_attack_fail_opposer
            self.state.mover.state.hp += self.hp_attack_fail_mover
            self.state.opposer.state.xp += self.xp_attack_fail_opposer
            self.state.mover.state.xp += self.xp_attack_fail_mover
            move = ("attack", False)
            self.state.moves.append(move)
            self.state.mover.state.moves[-1].append(move)

    def _retreat(self):
        if random.random() < self.prob_retreat_success:
            self.state.opposer.state.hp += self.hp_retreat_success_opposer
            self.state.mover.state.hp += self.hp_retreat_success_mover
            self.state.opposer.state.xp += self.xp_retreat_success_opposer
            self.state.mover.state.xp += self.xp_retreat_success_mover
            move = ("retreat", True)
            self.state.moves.append(move)
            self.state.mover.state.moves[-1].append(move)
        else:
            self.state.opposer.state.hp += self.hp_retreat_fail_opposer
            self.state.mover.state.hp += self.hp_retreat_fail_mover
            self.state.opposer.state.xp += self.xp_retreat_fail_opposer
            self.state.mover.state.xp += self.xp_retreat_fail_mover
            move = ("retreat", False)
            self.state.moves.append(move)
            self.state.mover.state.moves[-1].append(move)


class Environment:
    """
    This class represents the environment in which the agents operate. It
    typically has a state that changes over time, based on the actions of the
    agents. The Environment class might have methods for updating the state,
    computing the reward for each agent, and visualizing the current state of
    the environment.
    """

    class State:
        def __init__(self):
            self.num_agents = 0
            self.num_rooms = 0
            self.num_rounds = 0
            self.game_rounds = []
            self.agents = []
            self.agents_fallen = []
            self.agents_respawned = []
            self.hp_by_strategy = []
            self.xp_by_strategy = []
            self.population_by_strategy = []

        @property
        def _new_agent_idx(self):
            idx = self.num_agents
            self.num_agents += 1
            return idx

        @property
        def _new_room_idx(self):
            idx = self.num_rooms
            self.num_rooms += 1
            return idx

        @property
        def _new_round_idx(self):
            idx = self.num_rounds
            self.num_rounds += 1
            self.num_rooms = 0
            return idx

        def update_counts(self, strategies):
            self._update_count_hp_by_strategy(strategies)
            self._update_count_xp_by_strategy(strategies)
            self._update_count_population_by_strategy(strategies)

        def _update_count_hp_by_strategy(self, strategies):
            hp_by_strategy = {strategy: 0 for strategy in strategies}
            for agent in self.agents[-1]:
                hp_by_strategy[agent.strategy.__name__] += agent.state.hp
            self.hp_by_strategy.append(hp_by_strategy)

        def _update_count_xp_by_strategy(self, strategies):
            xp_by_strategy = {strategy: 0 for strategy in strategies}
            for agent in self.agents[-1]:
                xp_by_strategy[agent.strategy.__name__] += agent.state.xp
            self.xp_by_strategy.append(xp_by_strategy)

        def _update_count_population_by_strategy(self, strategies):
            population_by_strategy = {}
            agent_strategies = [a.strategy.__name__ for a in self.agents[-1]]
            for strategy in strategies:
                population_by_strategy[strategy] = agent_strategies.count(strategy)
            self.population_by_strategy.append(population_by_strategy)

    def __init__(self, num_rounds, agent_properties, game_properties):
        self.num_rounds = num_rounds
        self.agent_properties = agent_properties
        self.game_properties = game_properties
        self.state = self.State()
        self.state.agents.append(
            [
                Agent(
                    idx=self.state._new_agent_idx,
                    init_hp=agent_properties["init_hp"],
                    strategy=strategy,
                )
                for strategy in agent_properties["strategies"].values()
                for _ in range(agent_properties["num_agents_per_strategy"])
            ]
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def run_game(self, game):
        game.run(self)

    def get_shuffled_pairs(self):
        latest_agents = self.state.agents[-1]
        random.shuffle(latest_agents)
        return [
            (latest_agents[i], latest_agents[i + 1])
            for i in range(0, len(latest_agents), 2)
        ]

    def choose_agent_strategy(self, strategies):
        xp_by_strategy = self.state.xp_by_strategy[-1]
        total_xp = sum(xp_by_strategy.values())
        strategy_choices = list(strategies.values())
        try:
            strategy_probabilities = {
                strategy: xp / total_xp for strategy, xp in xp_by_strategy.items()
            }
            if random.random() < self.game_properties["prob_strategy_choice_mutation"]:
                strategy_weights = [1 / s for s in strategy_probabilities.values()]
            else:
                strategy_weights = list(strategy_probabilities.values())
            return random.choices(strategy_choices, weights=strategy_weights, k=1)[0]
        except ZeroDivisionError as e:
            log.error(e)
            return random.choices(strategy_choices, k=1)[0]

    def simulate_rounds(self):
        for _ in range(self.num_rounds):
            self.simulate_round()

    def simulate_round(self):
        round_idx = self.state._new_round_idx
        games = [
            Game(
                round_idx=round_idx,
                room_idx=self.state._new_room_idx,
                agents=pair,
                **self.game_properties,
            )
            for pair in self.get_shuffled_pairs()
        ]
        self.state.game_rounds.append(games)
        threads = [
            threading.Thread(target=self.run_game, args=(game,)) for game in games
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        latest_agents = self.state.agents[-1]
        agents_survived = [agent for agent in latest_agents if agent.state.hp >= 0]
        agents_fallen = list(
            set(latest_agents).symmetric_difference(set(agents_survived))
        )
        self.state.update_counts(self.agent_properties["strategies"].keys())
        agents_respawned = [
            Agent(
                idx=self.state._new_agent_idx,
                init_hp=self.agent_properties["init_hp"],
                strategy=self.choose_agent_strategy(
                    self.agent_properties["strategies"]
                ),
            )
            for _ in range(len(agents_fallen))
        ]
        self.state.agents.append(agents_survived + agents_respawned)
        self.state.agents_fallen.append(agents_fallen)
        self.state.agents_respawned.append(agents_respawned)


def main(save_path=None):
    env = Environment(
        num_rounds=1000,
        agent_properties=dict(
            init_hp=100,
            strategies={
                s.__name__: s
                for s in [
                    AttackStrategy,
                    RetreatStrategy,
                    RetaliateStrategy,
                    ExploitStrategy,
                ]
            },
            num_agents_per_strategy=250,
        ),
        game_properties=dict(
            prob_skip=0.2,
            prob_attack_success=0.8,
            prob_retreat_success=0.4,
            prob_strategy_choice_mutation=0.01,
            hp_attack_success_opposer=-10,
            hp_attack_success_mover=0,
            hp_attack_fail_opposer=0,
            hp_attack_fail_mover=0,
            hp_retreat_success_opposer=0,
            hp_retreat_success_mover=0,
            hp_retreat_fail_opposer=0,
            hp_retreat_fail_mover=0,
            xp_attack_success_opposer=0,
            xp_attack_success_mover=7,
            xp_attack_fail_opposer=2,
            xp_attack_fail_mover=0,
            xp_retreat_success_opposer=0,
            xp_retreat_success_mover=43,
            xp_retreat_fail_opposer=0,
            xp_retreat_fail_mover=0,
            hp_bonus=-1,
            xp_bonus=2,
            is_hp_reset=False,
            is_xp_reset=False,
        ),
    )
    env.simulate_rounds()
    if save_path:
        env.save(save_path)


if __name__ == "__main__":
    main(save_path="/tmp/environment.pickle")
