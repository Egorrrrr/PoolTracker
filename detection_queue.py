import queue
from threading import Thread

import numpy

import detect


class DetectionQueue:

    def __init__(self):
        self.is_running = False
        self.thread = Thread(target=self.run)
        self.frame_queue = queue.Queue(maxsize=1)
        self.ready_frames = queue.LifoQueue(maxsize=1)

    def put(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get(self):
        if not self.ready_frames.empty():
            return self.ready_frames.get_nowait()
        return None

    def start(self):
        self.is_running = True;
        self.thread.start()

    def run(self):
        while self.is_running:
            frame = self.frame_queue.get()
            results = detect.detect(frame)
            out = (results, frame)
            self.ready_frames.put(out)
