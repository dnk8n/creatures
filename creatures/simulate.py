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

    def __init__(self, env, game, agent):
        """
        Initializes a new instance of the Strategy class.

        Args:
            env: The instance of the Environment.
            game: The instance of the Game.
            agent: The instance of the Agent.
        """
        self.env: Environment = env
        self.game: Game = game
        self.agent: Agent = agent

    def action(self):
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

    def action(self):
        """
        Returns:
            "retreat"
        """
        return "retreat"


class AlwaysAttackStrategy(Strategy):
    """
    A strategy that always returns "attack".
    """

    def action(self):
        """
        Returns:
            "attack"
        """
        return "attack"


class TitForTatStrategy(Strategy):
    """
    A strategy that returns the opponent's last non-"skip" move, or "retreat"
    if the opponent has not made a move yet.

    TODO: Add memory of the past and knowledge of games the agent did not
    compete in.
    """

    def action(self):
        """
        Returns:
            The opponent's last non-"skip" move, or "retreat" if the opponent
            has not made a move yet.
        """
        return self.game.state.last_non_skip_move or "retreat"


class RetreatIfKnownAttackerStrategy(Strategy):
    """
    A strategy that returns "retreat" if the opponent's last move was "attack",
    or "attack" otherwise.
    """

    def action(self):
        """
        Returns:
            "retreat" if the opponent's last move was "attack", or "attack"
            otherwise.
        """
        if self.game.state.last_non_skip_move == "attack":
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
        def __init__(self, hp):
            self.hp = hp

    def __init__(
        self,
        idx: str,
        init_hp: int,
        strategy: "Strategy",
    ):
        self.idx = idx
        self.strategy = strategy
        self.state = self.State(hp=init_hp)


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
        survival_reward: int,
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
        self.survival_reward = survival_reward
        self.state = self.State()

    def run(self):
        time.sleep(random.random() * 5)

    # TODO: agents[0/1].strategy.action()

    def record_action(self, agent):
        if not self._skip_turn:
            self.state.moves.append()

    def submit_retreat(self, agent):
        if not self._skip_turn:
            self.state.moves.append(self.retreat)

    # def set_payoff_matrix(self, matrix):
    #     # Set the payoff matrix for the game
    #     pass

    # def compute_payoffs(self, actions):
    #     # Compute the payoffs for each agent based on the actions they took
    #     pass

    # def visualize(self):
    #     # Visualize the game matrix
    #     pass

    def _skip_turn(self):
        if random.random() < self.probability_skip:
            self._submit_skip
            return True
        else:
            return False

    def _submit_skip(self):
        self.state.moves.append(self.skip)

    def _attack(self, agent):
        if random.random() < self.probability_attack_success:
            agent.opponent.hp -= self.damage_attack_success_given
            self.hp -= self.damage_attack_success_taken
        else:
            agent.opponent.hp -= self.damage_attack_fail_given
            self.hp -= self.damage_attack_fail_taken

    def _retreat(self, agent):
        if random.random() < self.probability_retreat_success:
            agent.opponent.hp -= self.damage_retreat_success_given
            self.hp -= self.damage_retreat_success_taken
        else:
            agent.opponent.hp -= self.damage_retreat_fail_given
            self.hp -= self.damage_retreat_fail_taken

    def _skip(self, agent):
        pass

    def _execute_move(self, agent):
        if random.random() < self.probability_skip:
            pass


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
            self.game_history = []

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

    def __init__(self):
        self.state = self.State()

    def update_state(self, actions):
        # Update the state of the environment based on the actions of the agents
        pass

    def compute_reward(self, agent: "Agent") -> float:
        # Compute the reward for the given agent based on the current state of the environment
        pass

    def visualize(self):
        # Visualize the current state of the environment
        pass

    @staticmethod
    def get_pairs(agents):
        random.shuffle(agents)
        pairs = [(agents[i], agents[i + 1]) for i in range(0, len(agents), 2)]
        return pairs

    @staticmethod
    def run_game(game):
        agent_a = game.agents[0]
        agent_b = game.agents[1]
        game_title = f"Round {game.round_idx} - Room {game.room_idx} - {agent_a.idx}({agent_a.strategy.__name__}) vs {agent_b.idx}({agent_b.strategy.__name__})"
        log.info(f"BEGIN: {game_title}")
        game.run()
        log.info(f"END:   {game_title}")

    def init_agents(self, init_hp, strategies, num_agents_per_strategy):
        return [
            Agent(idx=self.state._new_agent_idx, init_hp=init_hp, strategy=strategy)
            for strategy in strategies
            for _ in range(num_agents_per_strategy)
        ]

    def simulate_round(self, round_idx, agents, game_properties):
        threads = [
            threading.Thread(
                target=self.run_game,
                args=(
                    Game(
                        round_idx=round_idx,
                        room_idx=self.state._new_room_idx,
                        agents=pair,
                        **game_properties,
                    ),
                ),
            )
            for pair in self.get_pairs(agents)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # TODO: Save results of games to environment history


def main():
    env = Environment()
    agents = env.init_agents(
        init_hp=100,
        strategies=[
            AlwaysAttackStrategy,
            AlwaysRetreatStrategy,
            TitForTatStrategy,
            RetreatIfKnownAttackerStrategy,
        ],
        num_agents_per_strategy=30,
    )
    game_properties = dict(
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
        survival_reward=1,
    )
    num_rounds = 3
    for _ in range(num_rounds):
        env.simulate_round(
            round_idx=env.state._new_round_idx,
            agents=agents,
            game_properties=game_properties,
        )


if __name__ == "__main__":
    main()
