basePath = "/content/drive/MyDrive/TPSMM-GFPGAN-main"
%cd {basePath}

import cv2

def calculate_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Print the frames per second
    print(f"Frames per second (fps) of the video: {fps}")

    # Release the video capture object
    cap.release()
    return fps

# Path to the video file
video_path = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/generated.mp4"  # Replace with the actual path to your video file

# Calculate fps of the video
calculate_fps(video_path)


import random
fps = 30
total_frames = 5 * fps
print(total_frames)

# Number of frames to select
num_frames_to_select = 15

# Randomly select 15 frame numbers
frame_numbers = random.sample(range(total_frames), num_frames_to_select)
print(frame_numbers)

# Install OpenCV
!pip install opencv-python-headless

import cv2
import os

# Function to extract frames from a video
def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Extract frames 4 and 5
    frame_numbers = random.sample(range(total_frames), num_frames_to_select)

    for frame_number in frame_numbers:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if ret:
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame {frame_number} saved successfully.")
        else:
            print(f"Error: Unable to read frame {frame_number}.")

    # Release the video capture object
    cap.release()

# Path to the video file
video_path = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/generated.mp4"  # Replace with the actual path to your video file

# Directory to save the extracted frames
output_dir = "original_frame"

# Extract frames from the video
extract_frames(video_path, output_dir)

import cv2
import os
import random

def extract_frames(video_path, output_dir, frame_numbers):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    for frame_number in frame_numbers:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if ret:
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Frame {frame_number} saved successfully.")
        else:
            print(f"Error: Unable to read frame {frame_number}.")

    # Release the video capture object
    cap.release()

# Usage
total_frames = 150
num_frames_to_select = 15
frame_numbers = random.sample(range(total_frames), num_frames_to_select)

video_path1 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/generated.mp4"  # Replace with the actual path to your video file
output_dir1 = "/content/drive/MyDrive/per/tpsm_frames"  # Replace with the actual path to your directory
extract_frames(video_path1, output_dir1, frame_numbers)

video_path2 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/enhanced.mp4"  # Replace with the actual path to your video file
output_dir2 = "/content/drive/MyDrive/per/gfpgan_frames"  # Replace with the actual path to your directory
extract_frames(video_path2, output_dir2, frame_numbers)


video_path2 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/enhanced2.mp4"  # Replace with the actual path to your video file
output_dir2 = "/content/drive/MyDrive/per/realesrgan_frames"  # Replace with the actual path to your directory
extract_frames(video_path2, output_dir2, frame_numbers)