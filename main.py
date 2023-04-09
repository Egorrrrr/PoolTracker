import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading
from faces import recognition
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

from faces import main_faces
from detection_queue import DetectionQueue

new_ids = []
old_ids = {}
faces = []
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("OpenCV Video Feed")
        self.master.geometry("960x360")
        self.master.resizable(0, 0)
        self.create_widgets()

    def create_widgets(self):

        self.video_area = tk.Canvas(self.master, width=640, height=360, bg="black")
        self.video_area.pack(side=tk.RIGHT)

        self.controls_area = tk.Frame(self.master)
        self.controls_area.pack(side=tk.LEFT, fill=tk.BOTH)

        self.start_button = tk.Button(self.controls_area, text="Start", command=self.start_video_feed)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.controls_area, text="Stop", command=self.stop_video_feed)
        self.stop_button.pack(pady=10)

        self.listbox = tk.Listbox(self.controls_area)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        self.is_running = False

    def start_video_feed(self):

        if self.is_running:
            self.paused = False
            return

        self.is_running = True
        self.paused = False

        threading.Thread(target=self.video_feed_loop).start()

    def video_feed_loop(self):
        det_queue = DetectionQueue()
        det_queue.start()
        thread = threading.Thread(target=main_faces.run)
        thread.start()

        cap = cv2.VideoCapture(r"D:\pooltf\vids\2tr.mp4")
        ret, frame = cap.read()
        det_queue.put(cv2.resize(frame, (1400, 700), fx=0, fy=0))
        time.sleep(5)
        numfr = 0
        while self.is_running:

            #pause
            while self.paused:
                time.sleep(1)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1400, 700), fx=0, fy=0)
            if ret:
                numfr += 1
                det_queue.put(frame)
                det_res = det_queue.get()
                if det_res is None:
                    continue
                data = det_res[0]
                frame = det_res[1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if data is None:
                    continue
                detections, trackers, lines, start_row, end_row = data
                frame_with_dets = frame.copy()

                if trackers is not None:
                    for tracker in trackers[:, 4]:
                        if int(tracker) not in old_ids:
                            old_ids[int(tracker)] = str(tracker) + recognition.current_face
                            recognition.current_face = ""
                    tracker_copy = trackers[:, 4]
                    tracker_copy = ([old_ids[int(i)] for i in tracker_copy])
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        frame_with_dets,
                        trackers[:, 0:4],
                        detections['detection_classes'],
                        detections['detection_scores'],
                        "D:/pooltf/third_model/label_map.pbtxt",
                        track_ids=tracker_copy,
                        use_normalized_coordinates=False,
                        max_boxes_to_draw=20,
                        min_score_thresh=0,
                        agnostic_mode=True)
                i = 0
                self.listbox.delete(0, tk.END)
                for _ in lines:
                    self.listbox.insert(tk.END, "Дорожка {}".format(i))
                    i += 1
                frame_with_dets = self.put_tracks(frame_with_dets, lines, end_row)

                cv2.rectangle(frame_with_dets, (0, start_row), (frame_with_dets.shape[1], end_row), (255, 255, 255), 3)
                frame_with_dets = cv2.resize(frame_with_dets, (640, 360))

                photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_dets))
                self.video_area.create_image(0, 0, image=photo, anchor=tk.NW)
                self.video_area.image = photo
                time.sleep(0.18)

    def stop_video_feed(self):
        self.paused = True

    def put_tracks(self, frame_with_dets, lines, end_row):

        for line in lines:
            cv2.line(frame_with_dets, (0, line), (frame_with_dets.shape[1], line), (0, 255, 0), 3)

        if len(self.listbox.curselection()) > 0:
            index = self.listbox.curselection()[0]
            alpha = 0.6
            if index == len(lines) - 1:
                shapes = np.zeros_like(frame_with_dets, np.uint8)

                cv2.rectangle(shapes, (0, lines[index]), (frame_with_dets.shape[1], end_row), (255, 255, 255),
                              cv2.FILLED)

                frame_with_dets = cv2.addWeighted(frame_with_dets, alpha, shapes, 1 - alpha, 0)

            else:
                shapes = np.zeros_like(frame_with_dets, np.uint8)
                cv2.rectangle(shapes, (0, lines[index]), (frame_with_dets.shape[1], lines[index + 1]),
                              (255, 255, 255), cv2.FILLED)
                frame_with_dets = cv2.addWeighted(frame_with_dets, alpha, shapes, 1 - alpha, 0)
        return frame_with_dets


root = tk.Tk()
app = Application(master=root)
app.mainloop()
