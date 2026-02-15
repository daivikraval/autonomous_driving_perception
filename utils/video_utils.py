import cv2

def get_camera_capture(camera_index=0):
    cap = cv2.VideoCapture("video.m4v")
    if not cap.isOpened():
        raise IOError(f"Cannot open camera with index {camera_index}")
    return cap
