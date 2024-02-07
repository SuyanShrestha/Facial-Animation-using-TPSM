# FORMULA
# PSNR=10log10(peakval2)/MSE

from math import log10, sqrt
#import cv2
#import numpy as np

def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def main():
  original = cv2.imread("/content/drive/MyDrive/TPSMM-GFPGAN-main/original_frame/frame_130.jpg")
  compressed = cv2.imread("/content/drive/MyDrive/TPSMM-GFPGAN-main/frame/frame_130.jpg", 1)
  original = cv2.resize(original, (256, 256))
  compressed = cv2.resize(compressed, (256, 256))
  value = PSNR(original, compressed)
  print(f"PSNR value is {value} dB")

if __name__ == "__main__":
	main()
	
import cv2
import os
import random
import numpy as np
from math import log10, sqrt

def PSNR(original, compressed):
    # Ensure the images have the same size
    compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal.
        return 100  # Therefore PSNR have no importance.
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def calculate_average_psnr(video_path1, video_path2, output_dir1, output_dir2, frame_numbers, gan_name):
    # Open the video files
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Unable to open video file.")
        return

    psnr_values = []

    for frame_number in frame_numbers:
        # Set the frame position
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Check if the frames are read successfully
        if ret1 and ret2:
            # Save the frames
            frame_path1 = os.path.join(output_dir1, f"frame_{frame_number}.jpg")
            frame_path2 = os.path.join(output_dir2, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path1, frame1)
            cv2.imwrite(frame_path2, frame2)
            # print(f"Frame {frame_number} saved successfully.")

            # Calculate PSNR and add it to the list
            psnr = PSNR(frame1, frame2)
            psnr_values.append(psnr)
        else:
            print(f"Error: Unable to read frame {frame_number}.")

    # Release the video capture objects
    cap1.release()
    cap2.release()

    # Calculate and print the average PSNR
    average_psnr = np.mean(psnr_values)
    print(f"Average PSNR of {gan_name}: {average_psnr} dB")

# Usage
total_frames = 150
num_frames_to_select = 15
frame_numbers = random.sample(range(total_frames), num_frames_to_select)

video_path1 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/generated.mp4"
output_dir1 = "/content/drive/MyDrive/per/tpsm_frames"
video_path2 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/enhanced.mp4"
output_dir2 = "/content/drive/MyDrive/per/gfpgan_frames"
gan_name = 'GFPGAN'
calculate_average_psnr(video_path1, video_path2, output_dir1, output_dir2, frame_numbers, gan_name)

video_path1 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/generated.mp4"
output_dir1 = "/content/drive/MyDrive/per/tpsm_frames"
video_path2 = "/content/drive/MyDrive/TPSMM-GFPGAN-main/outputs/enhanced2.mp4"
output_dir2 = "/content/drive/MyDrive/per/realesrgan_frames"
gan_name = 'Real-ESRGAN'
calculate_average_psnr(video_path1, video_path2, output_dir1, output_dir2, frame_numbers, gan_name)