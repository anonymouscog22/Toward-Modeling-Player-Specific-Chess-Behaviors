"""Tournament simulation utilities for head-to-head competitions.

This module implements several tournament managers (Single Elimination,
Round Robin, Swiss System) that orchestrate match scheduling, execution and
standings computation using the `run_match_series` helper to simulate games.
"""

import itertools
import random
from typing import List, Optional

from src.core.config import Config
from src.core.utils import getLogger
from src.evaluation.match import run_match_series
from src.models.maia import MaiaEngine

logger = getLogger()


class MatchNode:
    def __init__(
        self,
        left: "Optional[MatchNode]" = None,
        right: "Optional[MatchNode]" = None,
        player: Optional[str] = None,
    ):
        self.left = left
        self.right = right
        self.player = player
        self.winner: Optional[str] = None


class TournamentManager:
    def __init__(
        self, engine: MaiaEngine, config: Config, players: List[str], num_games: int = 2
    ):
        self.engine = engine
        self.config = config
        self.players = players
        self.num_games = num_games

    def run_tournament(self):
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _determine_winner(
        self, player_a: str, player_b: str, results: List[str]
    ) -> str:
        score_a, score_b = 0.0, 0.0
        for i, res in enumerate(results):
            if res == "1/2-1/2":
                score_a += 0.5
                score_b += 0.5
            elif res == "1-0":
                score_a += 1 if i % 2 == 0 else 0
                score_b += 0 if i % 2 == 0 else 1
            elif res == "0-1":
                score_b += 1 if i % 2 == 0 else 0
                score_a += 0 if i % 2 == 0 else 1

        if score_a > score_b:
            return player_a
        elif score_b > score_a:
            return player_b
        return random.choice([player_a, player_b])


class SingleElimination(TournamentManager):
    def __init__(
        self, engine: MaiaEngine, config: Config, players: List[str], num_games: int = 2
    ):
        super().__init__(engine, config, players, num_games)
        random.shuffle(self.players)
        self.root = self._build_tree(self.players)

    def _build_tree(self, players: List[str]) -> MatchNode:
        if len(players) == 1:
            return MatchNode(player=players[0])
        mid = len(players) // 2
        return MatchNode(
            left=self._build_tree(players[:mid]), right=self._build_tree(players[mid:])
        )

    def _play_match(self, player_a: str, player_b: str) -> str:
        logger.info(f"Scheduled Matchup: {player_a} vs {player_b}")
        results = run_match_series(
            self.engine, self.config, player_a, player_b, self.num_games
        )
        winner = self._determine_winner(player_a, player_b, results)
        logger.info(f"Matchup concluded. Victory: {winner}")
        return winner

    def _resolve(self, node: MatchNode) -> str:
        if node.player:
            return node.player
        assert node.left is not None and node.right is not None

        player_a = self._resolve(node.left)
        player_b = self._resolve(node.right)
        node.winner = self._play_match(player_a, player_b)

        logger.info("=" * 50)
        logger.info(f"Tournament bracket updated. Advancing player: {node.winner}")
        self.display_bracket(self.root)
        logger.info("=" * 50 + "\n")
        return node.winner

    def display_bracket(
        self,
        node: MatchNode,
        prefix: str = "",
        is_left: bool = True,
        is_root: bool = True,
    ):
        if node is None:
            return
        if node.right:
            new_prefix = prefix + ("" if is_root else ("│   " if is_left else "    "))
            self.display_bracket(node.right, new_prefix, False, False)

        name = (
            f"{node.winner} (Winner)"
            if node.winner
            else (node.player if node.player else "[Pending Match]")
        )

        if is_root:
            logger.info(f"{name}")
        else:
            indicator = "└── " if is_left else "┌── "
            logger.info(f"{prefix}{indicator}{name}")

        if node.left:
            new_prefix = prefix + ("" if is_root else ("    " if is_left else "│   "))
            self.display_bracket(node.left, new_prefix, True, False)

    def run_tournament(self) -> str:
        logger.info("Initial Tournament Bracket:")
        self.display_bracket(self.root)
        logger.info("=" * 50 + "\n")
        champion = self._resolve(self.root)
        logger.info(f"Tournament concluded. Grand Champion: {champion}\n")
        return champion


class RoundRobin(TournamentManager):
    def __init__(
        self, engine: MaiaEngine, config: Config, players: List[str], num_games: int = 2
    ):
        super().__init__(engine, config, players, num_games)
        self.scores = {player: 0.0 for player in self.players}

    def run_tournament(self) -> str:
        logger.info("=" * 50)
        logger.info("Starting Round Robin tournament")
        logger.info("=" * 50 + "\n")

        matchups = list(itertools.combinations(self.players, 2))
        for player_a, player_b in matchups:
            logger.info(f"Scheduled match: {player_a} vs {player_b}")
            results = run_match_series(
                self.engine, self.config, player_a, player_b, self.num_games
            )

            for i, res in enumerate(results):
                if res == "1/2-1/2":
                    self.scores[player_a] += 0.5
                    self.scores[player_b] += 0.5
                elif res == "1-0":
                    self.scores[player_a] += 1.0 if i % 2 == 0 else 0
                    self.scores[player_b] += 0 if i % 2 == 0 else 1.0
                elif res == "0-1":
                    self.scores[player_b] += 1.0 if i % 2 == 0 else 0
                    self.scores[player_a] += 0 if i % 2 == 0 else 1.0

        standings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("=" * 50)
        logger.info("Final Round Robin Standings:")
        for rank, (player, score) in enumerate(standings, 1):
            logger.info(f"{rank}. {player} with {score} points")
        logger.info("=" * 50 + "\n")

        champion = standings[0][0]
        logger.info(f"The undisputed champion is {champion}!\n")
        return champion


class SwissSystem(TournamentManager):
    def __init__(
        self,
        engine: MaiaEngine,
        config: Config,
        players: List[str],
        num_games: int = 2,
        num_rounds: int = 3,
    ):
        super().__init__(engine, config, players, num_games)
        self.num_rounds = num_rounds
        self.scores = {player: 0.0 for player in self.players}
        self.played_pairs = set()
        self.byes = (
            set()
        )  # History of players who have received a bye to avoid duplicates

    def _get_pairings(self) -> tuple[List[tuple[str, str]], Optional[str]]:
        """Generate pairings for the round and designate a bye player if required.

        Returns a tuple (pairings, bye_player) where `pairings` is a list of
        (player_a, player_b) tuples and `bye_player` is the identifier of the
        player receiving a bye when the number of participants is odd.
        """
        sorted_players = sorted(
            self.scores.keys(), key=lambda p: self.scores[p], reverse=True
        )
        pairings = []
        used = set()
        bye_player = None

        # Handle bye assignment when the number of players is odd. Select the
        # lowest-ranked player (by score) who has not yet received a bye.
        if len(sorted_players) % 2 != 0:
            for p in reversed(sorted_players):
                if p not in self.byes:
                    bye_player = p
                    self.byes.add(p)
                    used.add(p)
                    break

            # Fallback: if all players have already received a bye, assign to the last player
            if bye_player is None:
                bye_player = sorted_players[-1]
                used.add(bye_player)

        for i in range(len(sorted_players)):
            p1 = sorted_players[i]
            if p1 in used:
                continue

            # Search for an available opponent that the player has not yet faced
            for j in range(i + 1, len(sorted_players)):
                p2 = sorted_players[j]
                match_id = tuple(sorted([p1, p2]))

                if p2 not in used and match_id not in self.played_pairs:
                    pairings.append((p1, p2))
                    self.played_pairs.add(match_id)
                    used.add(p1)
                    used.add(p2)
                    break
            else:
                # Fallback: pair with the next available player if all nearby opponents have been played
                for j in range(i + 1, len(sorted_players)):
                    p2 = sorted_players[j]
                    if p2 not in used:
                        pairings.append((p1, p2))
                        used.add(p1)
                        used.add(p2)
                        break

        return pairings, bye_player

    def run_tournament(self) -> str:
        logger.info("=" * 50)
        logger.info(f"Starting Swiss System tournament ({self.num_rounds} rounds)")
        logger.info("=" * 50 + "\n")

        for round_num in range(1, self.num_rounds + 1):
            logger.info(f"--- Round {round_num} ---")
            pairings, bye_player = self._get_pairings()

            # Award the automatic point for the bye
            if bye_player:
                logger.info(
                    "[BYE] %s receives a bye (1.0 point) this round.", bye_player
                )
                self.scores[bye_player] += 1.0

            # Execute scheduled matches
            for player_a, player_b in pairings:
                logger.info(f"Scheduled match: {player_a} vs {player_b}")
                results = run_match_series(
                    self.engine, self.config, player_a, player_b, self.num_games
                )

                for i, res in enumerate(results):
                    if res == "1/2-1/2":
                        self.scores[player_a] += 0.5
                        self.scores[player_b] += 0.5
                    elif res == "1-0":
                        self.scores[player_a] += 1.0 if i % 2 == 0 else 0
                        self.scores[player_b] += 0 if i % 2 == 0 else 1.0
                    elif res == "0-1":
                        self.scores[player_b] += 1.0 if i % 2 == 0 else 0
                        self.scores[player_a] += 0 if i % 2 == 0 else 1.0

            # Display intermediate standings after the round
            standings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
            logger.info("Standings after Round %d:", round_num)
            for rank, (player, score) in enumerate(standings, 1):
                logger.info(f"{rank}. {player} with {score} points")
            logger.info("\n")

        # Classement final
        standings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        logger.info("=" * 50)
        logger.info("Final Swiss System Standings:")
        for rank, (player, score) in enumerate(standings, 1):
            logger.info(f"{rank}. {player} with {score} points")
        logger.info("=" * 50 + "\n")

        champion = standings[0][0]
        logger.info(f"The winner of the Swiss tournament is {champion}!\n")
        return champion
