import random
import threading
import time
from typing import List, Tuple

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
    A strategy that returns the opponent's last non-"skip" move, or "retreat"
    if the opponent has not made a move yet.
    """

    def action(env: "Environment" = None, game: "Game" = None, agent: "Agent" = None):
        """
        Returns:
            The opponent's last non-"skip" move, or "retreat" if the opponent
            has not made a move yet.
        """
        # TODO: Add memory of the past and knowledge of other games
        return game.state.last_non_skip_move or "retreat"


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
        damage_attack_success_given: int,
        damage_attack_success_taken: int,
        damage_attack_fail_given: int,
        damage_attack_fail_taken: int,
        damage_retreat_success_given: int,
        damage_retreat_success_taken: int,
        damage_retreat_fail_given: int,
        damage_retreat_fail_taken: int,
        probability_skip: float,
        probability_attack_success: float,
        probability_retreat_success: float,
        next_round_hp_bonus: int,
        next_round_xp_bonus: int,
    ):
        self.round_idx = round_idx
        self.room_idx = room_idx
        self.agents = agents
        self.damage_attack_success_given = damage_attack_success_given
        self.damage_attack_success_taken = damage_attack_success_taken
        self.damage_attack_fail_given = damage_attack_fail_given
        self.damage_attack_fail_taken = damage_attack_fail_taken
        self.damage_retreat_success_given = damage_retreat_success_given
        self.damage_retreat_success_taken = damage_retreat_success_taken
        self.damage_retreat_fail_given = damage_retreat_fail_given
        self.damage_retreat_fail_taken = damage_retreat_fail_taken
        self.probability_skip = probability_skip
        self.probability_attack_success = probability_attack_success
        self.probability_retreat_success = probability_retreat_success
        self.next_round_hp_bonus = next_round_hp_bonus
        self.next_round_xp_bonus = next_round_xp_bonus
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
                agent.state.hp += self.next_round_hp_bonus
                agent.state.xp += self.next_round_xp_bonus
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
            self.state.attacker.state.hp -= self.damage_attack_success_taken
            self.state.attacker.state.xp += self.damage_attack_success_given
            self.state.defender.state.hp -= self.damage_attack_success_given
            self.state.moves.append(
                (
                    "attack",
                    True,
                )
            )
        else:
            self.state.attacker.state.hp -= self.damage_attack_fail_taken
            self.state.attacker.state.xp += self.damage_attack_fail_given
            self.state.defender.state.hp -= self.damage_attack_fail_given
            self.state.moves.append(
                (
                    "attack",
                    False,
                )
            )

    def _retreat(self):
        if random.random() < self.probability_retreat_success:
            self.state.attacker.state.hp -= self.damage_retreat_success_taken
            self.state.attacker.state.xp += self.damage_retreat_success_given
            self.state.defender.state.hp -= self.damage_retreat_success_given
            self.state.moves.append(
                (
                    "retreat",
                    True,
                )
            )
        else:
            self.state.attacker.state.hp -= self.damage_retreat_fail_taken
            self.state.attacker.state.xp += self.damage_retreat_fail_given
            self.state.defender.state.hp -= self.damage_retreat_fail_given
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
            self.agents = []
            self.agents_fallen = []
            self.total_hp_by_strategy = None

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
            strategy_probabilities = {
                strategy: xp / total_xp
                for strategy, xp in self.total_xp_by_strategy.items()
            }
            strategy_choices = list(strategy_probabilities.keys())
            strategy_weights = list(strategy_probabilities.values())
            return random.choices(strategy_choices, weights=strategy_weights, k=1)[0]

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


def main():
    env = Environment(
        agent_properties=dict(
            init_hp=100,
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
            probability_attack_success=0.4,
            probability_retreat_success=0.8,
            damage_attack_success_given=4,
            damage_attack_success_taken=1,
            damage_attack_fail_given=0,
            damage_attack_fail_taken=2,
            damage_retreat_success_given=1,
            damage_retreat_success_taken=0,
            damage_retreat_fail_given=0,
            damage_retreat_fail_taken=3,
            next_round_hp_bonus=1,
            next_round_xp_bonus=4,
        ),
    )
    num_rounds = 1000
    for _ in range(num_rounds):
        env.simulate_round()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
