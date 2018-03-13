import cv2
import cozmo
import numpy as np
import os.path
from hackmob.gui import GUI

class HackApp:

    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    robot = None
    frame = None

    def __init__(self):

        self.frame = np.zeros((GUI.WINDOW_HEIGHT, GUI.WINDOW_WIDTH))

    def start(self, robot: cozmo.robot.Robot):

        self.robot = robot
        robot.camera.image_stream_enabled = True
        self.execute()

    def execute(self):

        while True:
            aux_frame = GUI.background_reload(self.robot)
            if aux_frame is not None:
                self.frame = aux_frame
                self.frame, faces_detected_n = GUI.widget_face_detection(self.frame)
                if faces_detected_n > 0:
                    self.frame = GUI.widgets_identity_reload(self.robot, self.frame)
            cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


app = HackApp()
cozmo.run_program(app.start)