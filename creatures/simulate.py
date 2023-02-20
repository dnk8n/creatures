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
        idx,
        init_hp,
        strategy,
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
            moves: A list of strings representing the moves made in the game so
                   far.
        """

        def __init__(self):
            """
            Initializes a new instance of the State class.
            """
            self.moves = []

        @property
        def last_move(self):
            """
            Returns:
                The last move made in the game, or None if no moves have been
                made yet.
            """
            if len(self.moves) == 0:
                return None
            return self.moves[-1]

        @property
        def last_non_skip_move(self):
            """
            Returns the last non-"skip" move made in the game, or None if no
            non-"skip" moves have been made

            Returns:
                The last non-"skip" move made in the game, or None if no
                non-"skip" moves have been made
            """
            moves = reversed(self.moves)
            try:
                return next(move for move in moves if move != "skip")
            except StopIteration:
                return None

    def __init__(
        self,
        agents: Tuple["Agent"],
        damage_attack_success_given: int,
        damage_attack_success_taken: int,
        damage_attack_fail_given: int,
        damage_attack_fail_taken: int,
        damage_retreat_success_given: int,
        damage_retreat_success_taken: int,
        damage_retreat_fail_given: int,
        damage_retreat_fail_taken: int,
        probability_freeze: float,
        probability_attack_success: float,
        probability_retreat_success: float,
        survival_reward: int,
    ):
        self.agents = agents
        self.damage_attack_success_given = damage_attack_success_given
        self.damage_attack_success_taken = damage_attack_success_taken
        self.damage_attack_fail_given = damage_attack_fail_given
        self.damage_attack_fail_taken = damage_attack_fail_taken
        self.damage_retreat_success_given = damage_retreat_success_given
        self.damage_retreat_success_taken = damage_retreat_success_taken
        self.damage_retreat_fail_given = damage_retreat_fail_given
        self.damage_retreat_fail_taken = damage_retreat_fail_taken
        self.probability_freeze = probability_freeze
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
        if random.random() < self.probability_freeze:
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
        if random.random() < self.probability_freeze:
            pass


class Environment:
    """
    This class represents the environment in which the agents operate. It
    typically has a state that changes over time, based on the actions of the
    agents. The Environment class might have methods for updating the state,
    computing the reward for each agent, and visualizing the current state of
    the environment.
    """

    _new_agent_idx = 0

    @property
    def new_agent_idx(self):
        Environment._new_agent_idx += 1
        return Environment._new_agent_idx

    class State:
        def __init__(self):
            self.game_history = []

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
    def run_game(game, round_idx):
        agent_a = game.agents[0]
        agent_b = game.agents[1]
        game_title = f"Round {round_idx} - {agent_a.idx}({agent_a.strategy.__name__}) vs {agent_b.idx}({agent_b.strategy.__name__})"
        log.info(f"BEGIN: {game_title}")
        game.run()
        log.info(f"END:   {game_title}")

    def run(self, num_rounds, num_agents_per_strategy, strategies, game_properties):
        agents = [
            Agent(idx=str(self.new_agent_idx).zfill(4), init_hp=100, strategy=strategy)
            for strategy in strategies
            for _ in range(num_agents_per_strategy)
        ]
        # Run the simulation for the specified number of rounds
        for round_idx in range(num_rounds):
            log.info(f"Round {round_idx} started!")
            threads = [
                threading.Thread(
                    target=self.run_game,
                    args=(Game(pair, **game_properties), round_idx),
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
    env.run(
        num_rounds=1000,
        num_agents_per_strategy=30,
        strategies=[
            AlwaysAttackStrategy,
            AlwaysRetreatStrategy,
            TitForTatStrategy,
            RetreatIfKnownAttackerStrategy,
        ],
        game_properties=dict(
            probability_freeze=0.1,
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
        ),
    )


if __name__ == "__main__":
    main()
