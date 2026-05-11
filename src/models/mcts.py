"""Monte Carlo Tree Search utilities used to select moves guided by a learned policy.

This module implements a compact Monte Carlo Tree Search (MCTS) that delegates
leaf evaluation and prior probability estimation to a provided child-generator
callable (typically a neural policy/value function). The implementation stores
visit counts and cumulative values at each node and applies a UCT-like selection
criterion combining exploitation (Q) and exploration (U).

The code is intentionally lightweight: expansion uses the provided policy to
instantiate child nodes whose prior probability exceeds a threshold, and the
value returned by the policy is used for backpropagation.
"""

from typing import Dict, Tuple

import chess
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    """MCTS node storing a prior probability, children and aggregated statistics.

    Attributes
    ----------
    maia_prob : float
        Prior probability produced by the policy for this node.
    children : dict[str, Node]
        Mapping from a UCI move string to the corresponding child node.
    visits : int
        Number of times the node has been visited during simulations.
    value : float
        Cumulative value (sum of backed-up values) associated with the node.
    """

    def __init__(self, maia_prob: float = 1.0) -> None:
        self.maia_prob: float = maia_prob
        self.children: Dict[str, "Node"] = {}
        self.visits: int = 0
        self.value: float = 0.0

    def compute_Q(self) -> float:
        """Return the mean value (Q) for this node, or 0 if unvisited."""
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def compute_U(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """Compute the exploration bonus U using a PUCT-like formula."""
        return c_puct * self.maia_prob * np.sqrt(parent_visits) / (1 + self.visits)

    def expand(
        self, child_generator, fen: str, activ_elo, opp_elo, threshold: float = 0.01
    ) -> float:
        """Expand the node using the provided child generator and return the position value.

        The `child_generator` is expected to return a tuple of the form
        (_, move_prob_dict, value), where `move_prob_dict` maps UCI move strings
        to prior probabilities and `value` is a scalar evaluation for the position.
        Children with prior probability greater than `threshold` are instantiated.
        """
        _, results, board_value = child_generator(fen, activ_elo, opp_elo)
        for move, prob in results.items():
            if prob > threshold:
                self.children[move] = Node(prob)
        return board_value


class MCTS:
    """A lightweight Monte Carlo Tree Search that uses an external policy/value.

    Parameters
    ----------
    child_generator : callable
        Callable accepting (fen, active_elo, opp_elo) and returning a tuple
        (_, move_probability_dict, value), where `move_probability_dict` is a
        mapping from UCI move strings to prior probabilities, and `value` is a
        scalar position evaluation used for backpropagation.
    """

    def __init__(self, child_generator) -> None:
        self.child_generator = child_generator
        self.root: Node = Node()

    def run(
        self,
        board: chess.Board,
        num_simulations: int,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        threshold: float = 0.01,
        activ_elo: int | str = 2500,
        opp_elo: int | str = 2500,
    ) -> Tuple[str, Dict[str, float]]:
        """Run the MCTS search and return the selected root move and root priors.

        Algorithmic steps:
         1. Expand the root node using the policy to create initial children.
         2. Repeat `num_simulations` times:
            - Selection: descend the tree selecting the child with maximal Q + U.
            - Expansion/Simulation: evaluate the leaf (using the child generator)
              and expand it if appropriate.
            - Backpropagation: update visit counts and cumulative values on the
              path from the leaf back to the root.
         3. Select the root child with the largest visit count as the best move.

        Returns
        -------
        best_root_move : str
            The UCI string of the move selected at the root.
        result : dict[str, float]
            Mapping from UCI move to the prior probability assigned at the root.
        """
        # 1. Initial root expansion using the provided policy/value generator.
        self.root.expand(
            self.child_generator, board.fen(), activ_elo, opp_elo, threshold
        )

        best_root_move = None

        for _ in range(num_simulations):
            current_node = self.root
            sim_board = board.copy()
            path = [current_node]

            # 2. Selection: descend the tree until a leaf node is reached.
            while current_node.children:
                best_score = -float("inf")
                best_move = None
                best_child = None

                for move, child in current_node.children.items():
                    score = child.compute_Q() + child.compute_U(
                        current_node.visits, c_puct=c_puct
                    )
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child

                assert best_move is not None, "No valid move found during tree descent"
                assert best_child is not None, "Best child is None during selection"

                sim_board.push_uci(best_move)
                current_node = best_child
                path.append(current_node)

            # 3. Simulation / Expansion: evaluate the leaf and possibly expand it.
            if sim_board.is_game_over():
                # Terminal positions are assigned a deterministic value:
                outcome = sim_board.outcome()
                if outcome is None or outcome.winner is None:
                    value = 0.0  # Draw
                else:
                    # A terminal loss for the side to move is represented as -1.0
                    value = -1.0
            else:
                # Non-terminal positions are evaluated by the external generator.
                value = current_node.expand(
                    self.child_generator, sim_board.fen(), activ_elo, opp_elo, threshold
                )

            # 4. Backpropagation: update statistics along the traversed path.
            for node in reversed(path):
                node.visits += 1
                node.value += value
                value = -value  # Negate value to account for alternating players

        # Select the root child with the maximal visit count as the final move.
        if not self.root.children:
            raise AssertionError(
                "Root node has no children after simulations; this indicates a failure in expansion or policy evaluation."
            )

        # Extract moves and their corresponding visit counts
        moves = list(self.root.children.keys())
        visit_counts = np.array(
            [child.visits for child in self.root.children.values()], dtype=np.float64
        )

        if temperature == 0:
            best_root_move = moves[np.argmax(visit_counts)]
        else:
            visits_with_temp = np.power(visit_counts, 1.0 / temperature)
            probabilities = visits_with_temp / np.sum(visits_with_temp)
            best_root_move = np.random.choice(moves, p=probabilities)

        result = {move: child.maia_prob for move, child in self.root.children.items()}
        return best_root_move, result
