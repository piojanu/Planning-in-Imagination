import pandas as pd


class ELOScoreboard(object):
    """Calculates and keeps players ELO statistics.

    Look at https://github.com/rshk/elo for reference
    """

    def __init__(self, players_ids, init_elo=1000):
        """Initialize ELO scoreboard.

        Args:
            players (list): List of players ids (weights file name will be fine).
            init_elo (float): Initial ELO of each player. (Default: 1000)
        """

        self.scores = pd.DataFrame(index=players_ids, columns=['elo'])
        self.scores.loc[:] = init_elo

    @staticmethod
    def load_csv(path):
        """Loads ELO scoreboard from .csv file.

        Args:
            path (str): Path to .csv file with data.

        Returns:
            ELOScoreboard: ELO scoreboard object with loaded data.
        """

        df = pd.read_csv(path, index_col=0, header=None)
        return ELOScoreboard(df.index, df.values)

    def save_csv(self, path):
        """Saves ELO scoreboard to .csv file.

        Args:
            path (str): Path to destination .csv file.
        """

        self.scores.to_csv(path, header=False)

    def update_rating(self, player, opponents, wins, draws, n_games=2, opponents_elo=None):
        """Update ELO rating of player after matches with opponents.

        Args:
            player (str): Player identifier.
            opponents (list of str): Opponents identifiers.
            wins (int), draws (int): Results of player in tournaments with opponents.
            n_games (int): Number of games played between each pair. (Default: 2)
            opponents_elo (list): ELO ratings of opponents. Ratings will be taken from scoreboard
        if None. (Default: None)
        """

        player_score = wins + .5 * draws
        player_elo = self.scores.loc[player, 'elo']

        expected_score = 0
        for idx, opponent in enumerate(opponents):
            opponent_elo = \
                self.scores.loc[opponent, 'elo'] if opponents_elo is None else opponents_elo[idx]
            expected_score += self._get_expected_score(player_elo, opponent_elo) * n_games

        self.scores.loc[player, 'elo'] = self._get_updated_elo(
            player_elo, expected_score, player_score)

    def plot(self, ax=None):
        """Plot players ELO ratings.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot in. If None, then plot on global axis.
        (Default: None)
        """

        import matplotlib.pyplot as plt

        self.scores.plot(ax=ax)
        plt.show()

    def _get_expected_score(self, A, B):
        """Calculate expected score of A in a match against B.

        Args:
            A (int): Elo rating for player A.
            B (int): Elo rating for player B.
        """

        return 1 / (1 + 10 ** ((B - A) / 400))

    def _get_updated_elo(self, old, exp, score, k=32):
        """Calculate the new Elo rating for a player.

        Args:
            old (int): The previous Elo rating.
            exp (float): The expected score for this match.
            score (float): The actual score for this match.
            k (int): The k-factor for Elo (default: 32).
        """

        return old + k * (score - exp)
