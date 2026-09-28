import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
import mediapipe as mp
from MediapipeHandGestureRecognition import MediapipeHandGestureRecognition as HandGesRec

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_top_gestures(image, Hand):
    for id_hand in range(Hand.num_detected_hands):
        pt_wrist = Hand.get_landmark(id_hand, Hand.WRIST)
        for rank in range(Hand.get_num_gestures(id_hand)):
            name = Hand.get_gesture(id_hand, rank=rank)
            score = Hand.get_score_gesture(id_hand, rank=rank)
            txt = '{}: {}({:#.2f})'.format(rank, name, score)
            pt_for_text = (pt_wrist[0]+10, pt_wrist[1]+30+30*rank)
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

    wname = 'MediaPipe HandGesture Top-3'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # raise max_results to see more than just the single most likely gesture
    Hand = HandGesRec(
        canned_gesture_classifier_options=mp.tasks.components.processors.ClassifierOptions(max_results=3)
    )

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
            draw_top_gestures(frame, Hand)

        cv2.imshow(wname, frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Hand.release()
    cap.release()

if __name__ == '__main__':
    main()
