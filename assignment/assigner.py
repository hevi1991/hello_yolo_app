from sklearn.cluster import KMeans


class Assigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_label = {}

    def get_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, box):
        # 裁剪
        image = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
        top_half_image = image[0 : int(image.shape[0] / 2), :]
        kmeans = self.get_model(top_half_image)

        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]

        bg_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - bg_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            box = player_detection["box"]
            player_color = self.get_player_color(frame, box)
            player_colors.append(player_color)

        # 训练
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
            
    def assign_team_id(self, frame, box, player_id):
        if player_id in self.player_team_label:
            return self.player_team_label[player_id]

        player_color = self.get_player_color(frame, box)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1


        if player_id == 96 or player_id == 146 or player_id == 132 or player_id == 155:
            team_id = 1
        self.player_team_label[player_id] = team_id
        
        return team_id
