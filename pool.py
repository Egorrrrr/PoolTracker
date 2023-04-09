
class Pool:
    def __init__(self, name):
        self.name = name
        self.coordinates = ()
        self.tracks = []
        self.swimmers = []


    def detect_pool(self):
        return True

    def detect_swimmers(self):
        return True

    def detect_tracks(self):
        return True