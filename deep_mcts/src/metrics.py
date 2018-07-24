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
    def _get_expected_score(A, B):
        """Calculate expected score of A in a match against B.

        Args:
            A (int): Elo rating for player A.
            B (int): Elo rating for player B.
        """

        return 1 / (1 + 10 ** ((B - A) / 400))

    @staticmethod
    def _get_updated_elo(old, exp, score, k=32):
        """Calculate the new Elo rating for a player.

        Args:
            old (int): The previous Elo rating.
            exp (float): The expected score for this match.
            score (float): The actual score for this match.
            k (int): The k-factor for Elo (default: 32).
        """

        return old + k * (score - exp)

    @staticmethod
    def calculate_update(p1_elo, p2_elo, p1_wins, p2_wins, draws):
        """Update ELO rating of two players after their matches.

        Args:
            p1_elo (float): Player one ELO.
            p2_elo (float): Player two ELO.
            p1_wins (int): Number of player one wins.
            p2_wins (int): Number of player two wins.
            draws (int): Number of draws between players.

        Return:
            float: Player one updated ELO rating.
            float: Player two updated ELO rating.
        """

        n_games = p1_wins + p2_wins + draws

        p1_score = p1_wins + .5 * draws
        p1_expected = ELOScoreboard._get_expected_score(p1_elo, p2_elo) * n_games

        p2_score = p2_wins + .5 * draws
        p2_expected = ELOScoreboard._get_expected_score(p2_elo, p1_elo) * n_games

        p1_updated = ELOScoreboard._get_updated_elo(p1_elo, p1_expected, p1_score)
        p2_updated = ELOScoreboard._get_updated_elo(p2_elo, p2_expected, p2_score)

        return p1_updated, p2_updated

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

    def update_player(self, player_id, opponents_elo, wins, draws, n_games=2):
        """Update ELO rating of player after matches with opponents.

        Args:
            player_id (str): Player identifier.
            opponents_elo (list of int): ELO ratings of opponent(s).
            wins (int): Number of player wins.
            draws (int): Number of draws between players.
            n_games (int): Number of games played between each pair. (Default: 2)
        """

        if not hasattr(opponents_elo, "__iter__"):
            opponents_elo = [opponents_elo, ]

        player_score = wins + .5 * draws
        player_elo = self.scores.loc[player_id, 'elo']

        expected_score = 0
        for opponent_elo in opponents_elo:
            expected_score += self._get_expected_score(player_elo, opponent_elo) * n_games

        updated_elo = self._get_updated_elo(player_elo, expected_score, player_score)
        self.scores.loc[player_id, 'elo'] = updated_elo

    def update_players(self, p1_id, p2_id, p1_wins, p2_wins, draws):
        """Update ELO rating of two players after their matches.

        Args:
            p1_id (str): Player one identifier.
            p2_id (str): Player two identifier.
            p1_wins (int): Number of player one wins.
            p2_wins (int): Number of player two wins.
            draws (int): Number of draws between players.
        """

        p1_elo = self.scores.loc[p1_id, 'elo']
        p2_elo = self.scores.loc[p2_id, 'elo']

        p1_updated, p2_updated = \
            self.calculate_update(p1_elo, p2_elo, p1_wins, p2_wins, draws)

        self.scores.loc[p1_id, 'elo'] = p1_updated
        self.scores.loc[p2_id, 'elo'] = p2_updated

    def plot(self, ax=None):
        """Plot players ELO ratings.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot in. If None, then plot on global axis.
        (Default: None)
        """

        import matplotlib.pyplot as plt

        self.scores.plot(ax=ax)
        plt.show()
