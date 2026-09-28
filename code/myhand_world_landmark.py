import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
from MediapipeHandLandmark import MediapipeHandLandmark as HandLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_index_finger_length(image, Hand):
    # get_landmark() is in pixels, so its scale changes with distance to the camera.
    # get_world_landmark() is in real-world meters, so it does not.
    for id_hand in range(Hand.num_detected_hands):
        pt_mcp = Hand.get_world_landmark(id_hand, Hand.INDEX_FINGER_MCP)
        pt_tip = Hand.get_world_landmark(id_hand, Hand.INDEX_FINGER_TIP)
        length_cm = np.linalg.norm(pt_tip - pt_mcp) * 100

        pt_tip_px = Hand.get_landmark(id_hand, Hand.INDEX_FINGER_TIP)
        txt = '{:#.1f} cm'.format(length_cm)
        pt_for_text = (pt_tip_px[0]+10, pt_tip_px[1])
        cv2.putText(image, org=pt_for_text, text=txt, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    wname = 'MediaPipe HandLandmark World Landmark'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    Hand = HandLmk()

    while cap.isOpened():
        frame_now = get_frame_number(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        results = Hand.detect(frame)

        if Hand.num_detected_hands > 0:
            draw_index_finger_length(frame, Hand)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
