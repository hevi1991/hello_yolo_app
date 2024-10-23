def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_width(box):
    x1, _, x2, _ = box
    return int(x2 - x1)

def get_distance(player_position, ball_center):
    return (
        ((player_position[0] - ball_center[0])**2 + (player_position[1] - ball_center[1])**2)**0.5
    )