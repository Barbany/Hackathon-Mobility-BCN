import cv2
import os.path
import numpy as np
import sqlite3

class GUI:

    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    FACES_DB_PATH = os.path.join(ROOT_PATH, 'res', 'hackmob.db')

    ROBOT_CAMERA_WIDTH = 320
    ROBOT_CAMERA_HEIGHT = 240

    WINDOW_WIDTH = ROBOT_CAMERA_WIDTH*3
    WINDOW_HEIGHT = ROBOT_CAMERA_HEIGHT*3
    WINDOW_PADDING = 20

    FACE_IDENTITY_BLOCK_WIDTH = 200
    FACE_IDENTITY_BLOCK_HEIGHT = 200
    FACE_IDENTITY_BLOCK_POS_X1 = WINDOW_WIDTH - WINDOW_PADDING - FACE_IDENTITY_BLOCK_WIDTH
    FACE_IDENTITY_BLOCK_POS_Y1 = WINDOW_HEIGHT - WINDOW_PADDING - FACE_IDENTITY_BLOCK_HEIGHT
    FACE_IDENTITY_BLOCK_POS_X2 = WINDOW_WIDTH - WINDOW_PADDING
    FACE_IDENTITY_BLOCK_POS_Y2 = WINDOW_HEIGHT - WINDOW_PADDING
    FACE_IDENTITY_BLOCK_COLOR = (255, 255, 255)
    FACE_IDENTITY_BLOCK_TRANSPARENCY = 0.85

    FACE_IDENTITY_IMAGE_WIDTH = 125
    FACE_IDENTITY_IMAGE_HEIGHT = 125
    FACE_IDENTITY_IMAGE_MARGIN_TOP = 20
    FACE_IDENTITY_IMAGE_POS_X1 = FACE_IDENTITY_BLOCK_POS_X1 + int((FACE_IDENTITY_BLOCK_WIDTH - FACE_IDENTITY_IMAGE_WIDTH) / 2)
    FACE_IDENTITY_IMAGE_POS_Y1 = FACE_IDENTITY_BLOCK_POS_Y1 + FACE_IDENTITY_IMAGE_MARGIN_TOP
    FACE_IDENTITY_IMAGE_POS_X2 = FACE_IDENTITY_IMAGE_POS_X1 + FACE_IDENTITY_IMAGE_WIDTH
    FACE_IDENTITY_IMAGE_POS_Y2 = FACE_IDENTITY_IMAGE_POS_Y1 + FACE_IDENTITY_IMAGE_HEIGHT

    FACE_IDENTITY_NAME_POS_X1 = None
    FACE_IDENTITY_NAME_POS_Y1 = None
    FACE_IDENTITY_NAME_MARGIN_TOP = 20
    FACE_IDENTITY_NAME_FONT_FACE = cv2.FONT_HERSHEY_PLAIN
    FACE_IDENTITY_NAME_FONT_SCALE = 1.2
    FACE_IDENTITY_NAME_COLOR = (0, 0, 0)

    FACE_DETECTION_MODEL_PATH_STRUCTURE = os.path.join(ROOT_PATH, 'models', 'face_detection', 'face_detection.prototxt')
    FACE_DETECTION_MODEL_PATH_WEIGHTS = os.path.join(ROOT_PATH, 'models', 'face_detection', 'face_detection.caffemodel')
    FACE_DETECTION_MODEL = cv2.dnn.readNetFromCaffe(FACE_DETECTION_MODEL_PATH_STRUCTURE, FACE_DETECTION_MODEL_PATH_WEIGHTS)

    @staticmethod
    def background_reload(robot):
        frame_im = robot.world.latest_image
        if frame_im is not None:
            frame = np.array(frame_im.raw_image)
            frame_res = cv2.resize(frame, (GUI.WINDOW_WIDTH, GUI.WINDOW_HEIGHT), interpolation=cv2.INTER_CUBIC)
            return frame_res
        else:
            return None

    @staticmethod
    def widgets_identity_compute_name_pos(text):
        text_size, _ = cv2.getTextSize(
            text=text,
            fontFace=GUI.FACE_IDENTITY_NAME_FONT_FACE,
            fontScale=GUI.FACE_IDENTITY_NAME_FONT_SCALE,
            thickness=1
        )
        GUI.FACE_IDENTITY_NAME_POS_X1 = GUI.FACE_IDENTITY_BLOCK_POS_X1 + int((GUI.FACE_IDENTITY_BLOCK_WIDTH - text_size[0]) / 2)
        GUI.FACE_IDENTITY_NAME_POS_Y1 = GUI.FACE_IDENTITY_IMAGE_POS_Y2 + GUI.FACE_IDENTITY_NAME_MARGIN_TOP + text_size[1]

    @staticmethod
    def widgets_identity_reload(robot, frame):

        # Recognize User
        user = GUI.widgets_face_identification(robot)
        user_visualized_name = user['name'] + ' ' + user['surnames']

        # Adds the background to the Identity Widget
        frame_copy = frame.copy()
        cv2.rectangle(
            img=frame_copy,
            pt1=(GUI.FACE_IDENTITY_BLOCK_POS_X1, GUI.FACE_IDENTITY_BLOCK_POS_Y1),
            pt2=(GUI.FACE_IDENTITY_BLOCK_POS_X2, GUI.FACE_IDENTITY_BLOCK_POS_Y2),
            color=GUI.FACE_IDENTITY_BLOCK_COLOR,
            thickness=-1
        )
        cv2.addWeighted(
            src1=frame_copy,
            alpha=GUI.FACE_IDENTITY_BLOCK_TRANSPARENCY,
            src2=frame,
            beta=1-GUI.FACE_IDENTITY_BLOCK_TRANSPARENCY,
            gamma=0,
            dst=frame
        )

        # Adds the image of the identified user to the Identity Widget
        identity_image_path = user['face_img']
        identity_image = cv2.resize(
            src=cv2.imread(identity_image_path),
            dsize=(GUI.FACE_IDENTITY_IMAGE_WIDTH, GUI.FACE_IDENTITY_IMAGE_HEIGHT)
        )
        frame[
            GUI.FACE_IDENTITY_IMAGE_POS_Y1:GUI.FACE_IDENTITY_IMAGE_POS_Y2,
            GUI.FACE_IDENTITY_IMAGE_POS_X1:GUI.FACE_IDENTITY_IMAGE_POS_X2, :] = identity_image

        # Adds the name of the identified user to the Identity Widget
        GUI.widgets_identity_compute_name_pos(user_visualized_name)
        cv2.putText(
            frame,
            text=user_visualized_name,
            org=(GUI.FACE_IDENTITY_NAME_POS_X1, GUI.FACE_IDENTITY_NAME_POS_Y1),
            fontFace=GUI.FACE_IDENTITY_NAME_FONT_FACE,
            fontScale=GUI.FACE_IDENTITY_NAME_FONT_SCALE,
            thickness=1,
            color=GUI.FACE_IDENTITY_NAME_COLOR
        )
        return frame

    @staticmethod
    def widget_face_detection(frame):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        GUI.FACE_DETECTION_MODEL.setInput(blob)
        detections = GUI.FACE_DETECTION_MODEL.forward()

        faces_detected_n = 0

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < 0.8:
                continue

            faces_detected_n += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        return frame, faces_detected_n

    @staticmethod
    def widgets_face_identification(robot):

        user = {
            'id': 0,
            'cozmo_id': 0,
            'face_img': os.path.join(GUI.ROOT_PATH, 'img', 'unknown.jpg'),
            'name': 'Unknown',
            'surnames': 'User',
            'age': 'Unknown'
        }

        try:
            print(robot.world.visible_face_count)
            if robot.world.visible_face_count() > 0:
                current_face_id = robot.world.visible_faces.__next__().face_id
                db_con = sqlite3.connect(GUI.FACES_DB_PATH)
                db_cursor = db_con.cursor()
                db_cursor.execute('SELECT * FROM users WHERE cozmo_id = {}'.format(current_face_id))
                db_res = db_cursor.fetchone()
                db_con.close()
                user = {
                    'id': db_res[0],
                    'cozmo_id': db_res[1],
                    'face_img': os.path.join(GUI.ROOT_PATH, 'img', db_res[2]),
                    'name': db_res[3],
                    'surnames': db_res[4],
                    'age': db_res[5]
                }
        finally:
            return user
