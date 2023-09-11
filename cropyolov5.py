from cv2 import waitKey
import torch
import cv2
import numpy as np
import time
import os

# Model
model_path = r"best.pt"  # Custom model path
video_path = r"fcccccccccccccc.mp4"  # Input video path
cpu_or_cuda = "cpu"  # Choose device; "cpu" or "cuda" (if cuda is available)
device = torch.device(cpu_or_cuda)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = model.to(device)
frame = cv2.VideoCapture(video_path)

frame_width = int(frame.get(3))
frame_height = int(frame.get(4))
size = (frame_width, frame_height)
writer = cv2.VideoWriter('output.mp4', -1, 8, size)

text_font = cv2.FONT_HERSHEY_PLAIN
color = (0, 0, 255)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0

# Create a folder to store cropped images
output_folder = 'cropped_images'
os.makedirs(output_folder, exist_ok=True)

# Inference Loop
while True:
    ret, image = frame.read()
    if ret:
        output = model(image)
        result = np.array(output.pandas().xyxy[0])
        for i in result:
            p1 = (int(i[0]), int(i[1]))
            p2 = (int(i[2]), int(i[3]))
            text_origin = (int(i[0]), int(i[1]) - 5)

            # Crop the detected object
            cropped_image = image[int(i[1]):int(i[3]), int(i[0]):int(i[2])]

            # Add a timestamp to the cropped image
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            filename = os.path.join(output_folder, f"{timestamp}.jpg")
            cv2.putText(cropped_image, timestamp, (10, 30), text_font, 1, (255, 255, 255), 2)

            # Save the cropped image with timestamp as the filename
            cv2.imwrite(filename, cropped_image)

            cv2.rectangle(image, p1, p2, color=color, thickness=2)
            cv2.putText(image, f"{i[-1]} {i[-3]:.2f}", org=text_origin,
                        fontFace=text_font, fontScale=text_font_scale,
                        color=color, thickness=2)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(image, fps, (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        writer.write(image)
        cv2.imshow("image", image)

    else:
        break

    if waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()

