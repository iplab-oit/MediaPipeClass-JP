import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
from MediapipeFaceLandmark import MediapipeFaceLandmark as FaceLmk

device = 0 # cameera device number

def get_frame_number(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * fps)
    return frame_now

def draw_top_blendshape(image, Face):
    for id_face in range(Face.num_detected_faces):
        name, score = Face.get_top_blendshape(id_face)
        pt_top = Face.get_landmark(id_face, 10)
        txt = '{}: {:#.2f}'.format(name, score)
        pt_for_text = (pt_top[0]+10, pt_top[1])
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

    wname = 'MediaPipe FaceLandmark Blendshape'
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    # make instance of our mediapipe class
    # blendshapes require output_face_blendshapes=True
    Face = FaceLmk(output_face_blendshapes=True)

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

        # Flip the image horizontally
        flipped_frame = cv2.flip(frame, 1) ### very important ####

        results = Face.detect(flipped_frame)

        if Face.num_detected_faces > 0:
            draw_top_blendshape(flipped_frame, Face)

        cv2.imshow(wname, flipped_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    Face.release()
    cap.release()

if __name__ == '__main__':
    main()
