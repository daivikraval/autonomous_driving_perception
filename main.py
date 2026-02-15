from models.lane_detection import detect_lanes
from models.object_detection import detect_objects
from utils.video_utils import get_camera_capture
import cv2

def main():
    cap = get_camera_capture(0)  # Default webcam

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('objectdetect.mp4', fourcc, 10.0, (frame_width, frame_height))

    display_width = 800
    display_height = 600

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, exiting.")
            break

        lane_frame = detect_lanes(frame)
        output_frame = detect_objects(lane_frame)

        out.write(output_frame)

        resized_frame = cv2.resize(output_frame, (display_width, display_height))
        cv2.imshow("Autonomous Driving Perception - Live", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
