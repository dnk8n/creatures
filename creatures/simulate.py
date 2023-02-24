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

    Subclasses must implement the `action` method, which calculates the next
    action to take based on the states of the environment, game, and agent.
    """

    @classmethod
    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Calculates the next action to take based on the states of the
        environment, game, and agent.

        Returns:
            The action to be taken in the current state.
        """
        pass


class AlwaysRetreatStrategy(Strategy):
    """
    A strategy that always returns "retreat".
    """

    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Returns:
            "retreat"
        """
        return "retreat"


class AlwaysAttackStrategy(Strategy):
    """
    A strategy that always returns "attack".
    """

    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Returns:
            "attack"
        """
        return "attack"


class TitForTatStrategy(Strategy):
    """
    A strategy that returns the opponent's last non-"skip" move, or "attack"
    if the opponent has not made a move yet.
    """

    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Returns:
            The opponent's last non-"skip" move, or "retreat" if the opponent
            has not made a move yet.
        """
        # TODO: Add memory of the past and knowledge of other games
        return game.state.last_non_skip_move or "attack"


class RetreatIfKnownAttackerStrategy(Strategy):
    """
    A strategy that returns "retreat" if the opponent's last move was "attack",
    or "attack" otherwise.
    """

    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Returns:
            "retreat" if the opponent's last move was "attack", or "attack"
            otherwise.
        """
        if game.state.last_non_skip_move == "attack":
            return "retreat"
        return "attack"


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

    def __init__(
        self,
        idx: str,
        init_hp: int,
        strategy: "Strategy",
    ):
        self.idx = idx
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
        """
        Represents the state of a game, including a list of moves made so far.

        Attributes:
            moves: A list of tuples representing the moves made in the game so
                   far, and their if successful.
        """

        def __init__(self):
            """
            Initializes a new instance of the State class.
            """
            self.moves: List[Tuple[str, bool]] = []
            self.attacker: "Agent" = None
            self.defender: "Agent" = None

        @property
        def last_move(self):
            """
            Returns:
                The last move made in the game, or None if no moves have been
                made yet.
            """
            if len(self.moves) == 0:
                return None
            return self.moves[-1][0]

        @property
        def last_non_skip_move(self):
            """
            Returns the last non-"skip" move made in the game, or None if no
            non-"skip" moves have been made

            Returns:
                The last non-"skip" move made in the game, or None if no
                non-"skip" moves have been made
            """
            try:
                return next(m for m, _ in reversed(self.moves) if m != "skip")
            except StopIteration:
                return None

    def __init__(
        self,
        round_idx: str,
        room_idx: str,
        agents: Tuple["Agent"],
        probability_skip: float,
        probability_attack_success: float,
        probability_retreat_success: float,
        hp_attack_success_defender: int,
        hp_attack_success_attacker: int,
        hp_attack_fail_defender: int,
        hp_attack_fail_attacker: int,
        hp_retreat_success_defender: int,
        hp_retreat_success_attacker: int,
        hp_retreat_fail_defender: int,
        hp_retreat_fail_attacker: int,
        xp_attack_success_defender: int,
        xp_attack_success_attacker: int,
        xp_attack_fail_defender: int,
        xp_attack_fail_attacker: int,
        xp_retreat_success_defender: int,
        xp_retreat_success_attacker: int,
        xp_retreat_fail_defender: int,
        xp_retreat_fail_attacker: int,
        hp_bonus: int,
        xp_bonus: int,
    ):
        self.round_idx = round_idx
        self.room_idx = room_idx
        self.agents = agents
        self.probability_skip = probability_skip
        self.probability_attack_success = probability_attack_success
        self.probability_retreat_success = probability_retreat_success
        self.hp_attack_success_defender = hp_attack_success_defender
        self.hp_attack_success_attacker = hp_attack_success_attacker
        self.hp_attack_fail_defender = hp_attack_fail_defender
        self.hp_attack_fail_attacker = hp_attack_fail_attacker
        self.hp_retreat_success_defender = hp_retreat_success_defender
        self.hp_retreat_success_attacker = hp_retreat_success_attacker
        self.hp_retreat_fail_defender = hp_retreat_fail_defender
        self.hp_retreat_fail_attacker = hp_retreat_fail_attacker
        self.xp_attack_success_defender = xp_attack_success_defender
        self.xp_attack_success_attacker = xp_attack_success_attacker
        self.xp_attack_fail_defender = xp_attack_fail_defender
        self.xp_attack_fail_attacker = xp_attack_fail_attacker
        self.xp_retreat_success_defender = xp_retreat_success_defender
        self.xp_retreat_success_attacker = xp_retreat_success_attacker
        self.xp_retreat_fail_defender = xp_retreat_fail_defender
        self.xp_retreat_fail_attacker = xp_retreat_fail_attacker
        self.hp_bonus = hp_bonus
        self.xp_bonus = xp_bonus
        self.state = self.State()

    def run(self):
        agent_a = self.agents[0]
        agent_b = self.agents[1]
        game_title = f"Round {self.round_idx} - Room {self.room_idx} - {agent_a.idx}({agent_a.strategy.__name__}) vs {agent_b.idx}({agent_b.strategy.__name__})"
        log.info(f"BEGIN: {game_title}")
        while not self._game_ended():
            if self.state.attacker == agent_a:
                self.state.attacker = agent_b
                self.state.defender = agent_a
            else:
                self.state.attacker = agent_a
                self.state.defender = agent_b
            self._process_turn()
        for agent in self.agents:
            if agent.state.hp >= 0:
                agent.state.hp += self.hp_bonus
                agent.state.xp += self.xp_bonus
        log.info(f"END:   {game_title}")

    def _game_ended(self):
        """
        Returns:
            True if the game has ended, False otherwise.
        """
        if not self.state.attacker or not self.state.defender:
            return False
        if self.state.attacker.state.hp < 0 or self.state.defender.state.hp < 0:
            return True
        if self.state.moves[-1][0] == "retreat" and self.state.moves[-1][1]:
            return True
        return False

    def _process_turn(self):
        if random.random() < self.probability_skip:
            self._skip()
        else:
            action = self.state.attacker.strategy.action(
                env=None, game=self, agent=self.state.attacker
            )
            if action == "attack":
                self._attack()
            elif action == "retreat":
                self._retreat()
            else:
                raise ValueError(f"Invalid Action, {action}")

    def _skip(self):
        self.state.moves.append(
            (
                "skip",
                None,
            )
        )

    def _attack(self):
        if random.random() < self.probability_attack_success:
            self.state.defender.state.hp += self.hp_attack_success_defender
            self.state.attacker.state.hp += self.hp_attack_success_attacker
            self.state.defender.state.xp += self.xp_attack_success_defender
            self.state.attacker.state.xp += self.xp_attack_success_attacker
            self.state.moves.append(
                (
                    "attack",
                    True,
                )
            )
        else:
            self.state.defender.state.hp += self.hp_attack_fail_defender
            self.state.attacker.state.hp += self.hp_attack_fail_attacker
            self.state.defender.state.xp += self.xp_attack_fail_defender
            self.state.attacker.state.xp += self.xp_attack_fail_attacker
            self.state.moves.append(
                (
                    "attack",
                    False,
                )
            )

    def _retreat(self):
        if random.random() < self.probability_retreat_success:
            self.state.defender.state.hp += self.hp_retreat_success_defender
            self.state.attacker.state.hp += self.hp_retreat_success_attacker
            self.state.defender.state.xp += self.xp_retreat_success_defender
            self.state.attacker.state.xp += self.xp_retreat_success_attacker
            self.state.moves.append(
                (
                    "retreat",
                    True,
                )
            )
        else:
            self.state.defender.state.hp += self.hp_retreat_fail_defender
            self.state.attacker.state.hp += self.hp_retreat_fail_attacker
            self.state.defender.state.xp += self.xp_retreat_fail_defender
            self.state.attacker.state.xp += self.xp_retreat_fail_attacker
            self.state.moves.append(
                (
                    "retreat",
                    False,
                )
            )


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
            self.total_hp_by_strategy = None
            self.total_xp_by_strategy = None

        @property
        def _new_agent_idx(self):
            idx = str(self.num_agents).zfill(4)
            self.num_agents += 1
            return idx

        @property
        def _new_room_idx(self):
            idx = str(self.num_rooms).zfill(4)
            self.num_rooms += 1
            return idx

        @property
        def _new_round_idx(self):
            idx = str(self.num_rounds).zfill(4)
            self.num_rounds += 1
            self.num_rooms = 0
            return idx

        def update_total_hp_by_strategy(self, strategies):
            total_hp_by_strategy = {strategy: 0 for strategy in strategies}
            for agent in self.agents:
                total_hp_by_strategy[agent.strategy] += agent.state.hp
            self.total_hp_by_strategy = total_hp_by_strategy

        def update_total_xp_by_strategy(self, strategies):
            total_xp_by_strategy = {strategy: 0 for strategy in strategies}
            for agent in self.agents:
                total_xp_by_strategy[agent.strategy] += agent.state.xp
            self.total_xp_by_strategy = total_xp_by_strategy

        def choose_agent_strategy(self):
            total_xp = sum(self.total_xp_by_strategy.values())
            try:
                strategy_probabilities = {
                    strategy: xp / total_xp
                    for strategy, xp in self.total_xp_by_strategy.items()
                }
                strategy_choices = list(strategy_probabilities.keys())
                strategy_weights = list(strategy_probabilities.values())
                return random.choices(strategy_choices, weights=strategy_weights, k=1)[
                    0
                ]
            except ZeroDivisionError:
                strategy_choices = list(strategy_probabilities.keys())
                return random.choices(strategy_choices, k=1)[0]

    def __init__(self, agent_properties, game_properties):
        self.agent_properties = agent_properties
        self.game_properties = game_properties
        self.state = self.State()
        self.state.agents = [
            Agent(
                idx=self.state._new_agent_idx,
                init_hp=agent_properties["init_hp"],
                strategy=strategy,
            )
            for strategy in agent_properties["strategies"]
            for _ in range(agent_properties["num_agents_per_strategy"])
        ]

    @staticmethod
    def run_game(game):
        game.run()

    def get_shuffled_pairs(self):
        random.shuffle(self.state.agents)
        return [
            (self.state.agents[i], self.state.agents[i + 1])
            for i in range(0, len(self.state.agents), 2)
        ]

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
        this_round_agents = list(self.state.agents)
        self.state.agents = [
            agent for agent in this_round_agents if agent.state.hp >= 0
        ]
        newly_fallen = list(
            set(this_round_agents).symmetric_difference(set(self.state.agents))
        )
        self.state.agents_fallen.extend(newly_fallen)
        self.state.update_total_hp_by_strategy(self.agent_properties["strategies"])
        self.state.update_total_xp_by_strategy(self.agent_properties["strategies"])
        self.state.agents.extend(
            [
                Agent(
                    idx=self.state._new_agent_idx,
                    init_hp=self.agent_properties["init_hp"],
                    strategy=self.state.choose_agent_strategy(),
                )
                for _ in range(len(newly_fallen))
            ]
        )


def main(save_to_file=None):
    env = Environment(
        agent_properties=dict(
            init_hp=1000,
            strategies=[
                AlwaysAttackStrategy,
                AlwaysRetreatStrategy,
                TitForTatStrategy,
                RetreatIfKnownAttackerStrategy,
            ],
            num_agents_per_strategy=30,
        ),
        game_properties=dict(
            probability_skip=0.1,
            probability_attack_success=0.7,
            probability_retreat_success=0.6,
            hp_attack_success_defender=-10,  # defender loses 10hp upon successful attack
            hp_attack_success_attacker=0,
            hp_attack_fail_defender=0,
            hp_attack_fail_attacker=0,
            hp_retreat_success_defender=0,
            hp_retreat_success_attacker=0,
            hp_retreat_fail_defender=0,
            hp_retreat_fail_attacker=-1,  # attacker loses 1hp if retreat fails
            xp_attack_success_defender=0,
            xp_attack_success_attacker=10,  # Attacker gains 10 xp upon successful attack
            xp_attack_fail_defender=1,  # Defender gains 1 xp if attack is a fail
            xp_attack_fail_attacker=0,
            xp_retreat_success_defender=1,  # Defender gains 1 xp if retreat is successful
            xp_retreat_success_attacker=0,
            xp_retreat_fail_defender=1,  # Defender gains 1xp if retreat is a fail
            xp_retreat_fail_attacker=0,
            hp_bonus=1,
            xp_bonus=1,
        ),
    )
    num_rounds = 1000
    for _ in range(num_rounds):
        env.simulate_round()
    if save_to_file:
        with open(save_to_file, "wb") as f:
            pickle.dump(env, f)


def load_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main(save_to_file="/tmp/environment.pickle")
