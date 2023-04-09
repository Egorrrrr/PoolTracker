class Swimmer:
    def __init__(self, coordinates_pool, coordinates_track, track, swimmer_id, name):
        self.swimmer_id = swimmer_id
        self.name = name
        self.coordinates_in_pool = coordinates_pool
        self.coordinates_track = coordinates_track
        self.track = track
