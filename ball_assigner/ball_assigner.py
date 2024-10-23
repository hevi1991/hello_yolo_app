from utils import get_center, get_distance


class BallAssigner:
    def __init__(self):
        self.max_player_and_ball_distance = 60

    def assign_ball_to_player(self, players, ball):
        ball_center = get_center(ball)

        minum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_position = player["box"]

            l_distance = get_distance(
                (player_position[0], player_position[-1]), ball_center
            )
            r_distance = get_distance(
                (player_position[2], player_position[-1]), ball_center
            )
            distance = min(l_distance, r_distance)

            if distance < self.max_player_and_ball_distance:
                if distance < minum_distance:
                    minum_distance = distance
                    assigned_player = player_id

        return assigned_player
