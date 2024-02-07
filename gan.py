from google.colab import drive
drive.mount('/content/drive')
basePath = "/content/drive/MyDrive/TPSMM-GFPGAN-main"
%cd {basePath}

tpsmmFolder = 'Thin-Plate-Spline-Motion-Model_main'
gfpganFolder = 'GFPGAN_main'
tpsmmPath = basePath + '/' + tpsmmFolder
gfpganPath = basePath + '/' + gfpganFolder

import torch
import os

# edit the config
device = torch.device('cuda:0')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
output_path = basePath + '/outputs'
source_image_path = basePath + '/inputs/source.png'
driving_video_path = basePath + '/inputs/driving.mp4'
output_video_path = basePath + '/outputs/generated.mp4'
config_path = tpsmmPath + '/config/vox-256.yaml'
checkpoint_path = tpsmmPath + '/checkpoints/vox.pth.tar'
predict_mode = 'relative' # ['standard', 'relative', 'avd']
find_best_frame = False # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

if not os.path.exists(output_path):
  os.makedirs(output_path)

pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
if(dataset_name == 'ted'): # for ted, the resolution is 384*384
    pixel = 384

if find_best_frame:
  !pip install face_alignment


try:
  import imageio
  import imageio_ffmpeg
except:
  !pip install imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import os

warnings.filterwarnings("ignore")

source_image = imageio.imread(source_image_path)
reader = imageio.get_reader(driving_video_path)

source_image = resize(source_image, (pixel, pixel))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


HTML(display(source_image, driving_video).to_html5_video())

%cd {tpsmmFolder}
from demo import load_checkpoints
inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)

from demo import make_animation
from skimage import img_as_ubyte
from demo import subprocess


if predict_mode=='relative' and find_best_frame:
    from demo import find_best_frame as _find
    i = _find(source_image, driving_video, device.type=='cpu')
    print ("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
else:
    predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = predict_mode)

#save resulting video
imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)



# Save resulting video with audio
output_video_path_with_audio = 'generated_withAudio.mp4'
audio_cmd = f'ffmpeg -i {output_video_path} -i {driving_video_path} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {output_video_path_with_audio}'
subprocess.call(audio_cmd, shell=True)
HTML(display(source_image, driving_video, predictions).to_html5_video())


from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width = 600):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

# output video
show_video(output_video_path_with_audio)

%cd ../GFPGAN_main

!pip install basicsr
!pip install facexlib # We use face detection and face restoration helper in the facexlib package
# Install other depencencies
!pip install -r requirements.txt
!python setup.py develop
!pip install realesrgan  # used for enhancing the background (non-face) regions
!wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P experiments/pretrained_models #Downloading Model GFPGANv1.4 / GFPGANv1.3 / GFPGANv1.2
%cd ..

import cv2
from tqdm import tqdm
from os import path

import os

inputVideoPath = output_path+'/generated.mp4'
unProcessedFramesFolderPath = output_path+'/frames'

if not os.path.exists(unProcessedFramesFolderPath):
  os.makedirs(unProcessedFramesFolderPath)

vidcap = cv2.VideoCapture(inputVideoPath)
numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps, "Frames: ", numberOfFrames)

for frameNumber in tqdm(range(numberOfFrames)):
    _,image = vidcap.read()
    cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4)+'.jpg'), image)


#Using GFPGAN version 1.4 and Real-ESRGAN for background
#If you don't want to enhance background then replace "realesrgan" with "None" in the below line
!python {gfpganFolder}/inference_gfpgan.py -i {unProcessedFramesFolderPath} -o {output_path} -v 1.4 -s 2 --only_center_face --bg_upsampler realesrgan

import os
restoredFramesPath = output_path + '/restored_imgs/'
processedVideoOutputPath = output_path

dir_list = os.listdir(restoredFramesPath)
dir_list.sort()

import cv2
import numpy as np

batch = 0
batchSize = 300
from tqdm import tqdm
for i in tqdm(range(0, len(dir_list), batchSize)):
  img_array = []
  start, end = i, i+batchSize
  print("processing ", start, end)
  for filename in  tqdm(dir_list[start:end]):
      filename = restoredFramesPath+filename;
      img = cv2.imread(filename)
      if img is None:
        continue
      height, width, layers = img.shape
      size = (width,height)
      img_array.append(img)


  out = cv2.VideoWriter(processedVideoOutputPath+'/batch_'+str(batch).zfill(4)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
  batch = batch + 1

  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()


concatTextFilePath = output_path + "/concat.txt"
concatTextFile=open(concatTextFilePath,"w")
for ips in range(batch):
  concatTextFile.write("file batch_" + str(ips).zfill(4) + ".avi\n")
concatTextFile.close()

finalVideoOutputPath = output_path + "/enhanced.mp4"
!ffmpeg -y -f concat -i {concatTextFilePath} -c copy {finalVideoOutputPath}

realesrganFolder = 'Real-ESRGAN_main'
realesrganPath = basePath + '/' + realesrganFolder

basePath = "/content/drive/MyDrive/TPSMM-GFPGAN-main"
%cd {basePath}

%cd ./Real-ESRGAN_main

!pip install basicsr
!pip install facexlib
!pip install gfpgan
!pip install ffmpeg-python
!pip install -r requirements.txt
!python setup.py develop
%cd ..

inputVideoFile = f'{output_path}/generated.mp4'
outputVideoFile = f'{output_path}/enhanced2.mp4'

from IPython.display import HTML
from base64 import b64encode
import os

os.rename(f'{output_path}/generated_out.mp4', outputVideoFile)

html_str=""
filepaths=[inputVideoFile, outputVideoFile]

for filepath in filepaths:
  width = 500
  mp4 = open(filepath,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  title = "Generated" if filepath == inputVideoFile else "Enhanced"
  html_str += f"""
  <div style="text-align: center;">
    <h1>{title}</h1>
    <video width={width} controls>
          <source src={data_url} type="video/mp4">
    </video>
  </div>
  """
HTML(f"""<div style="display:flex; justify-content:space-evenly">{html_str}</div>""")

