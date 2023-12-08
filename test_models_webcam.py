from threading import Thread
import cv2

from src.tensorflow_run import (
    load_Centernet_hourglass,
    create_library_for_labels,
    run_centernet,
)


def start_webcam():
    detector = load_Centernet_hourglass()
    category_index = create_library_for_labels()
    cap = cv2.VideoCapture(0)
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open DroidCam.")
        exit()
    while True:
        # Read a frame from the DroidCam
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            print("Error: Could not read frame.")
            break
        image_with_detections = run_centernet(frame, detector, category_index)
        # Display the frame
        cv2.imshow("DroidCam", image_with_detections)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the DroidCam and close the window
    cap.release()
    cv2.destroyAllWindows()


def main():
    thread = Thread(target=start_webcam())
    thread.start()
    thread.join()
    print("Success")
    return


if __name__ == "__main__":
    main()
