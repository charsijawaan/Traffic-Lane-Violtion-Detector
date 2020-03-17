import subprocess
import cv2

videoPath = 'video8.mp4'
videoOutputPath = 'test.mp4'

video = cv2.VideoCapture(videoPath)
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
origDuration = frame_count/fps
print(int(origDuration))

maxMult = 1;
minMult = 0.01;

newDuration = (origDuration - 5) / (10 - 5)
print('new duration = ' + str(newDuration))

newMult = str((newDuration - 5) / (10 - 5))
print('new mult = ' + newMult)


c = 'ffmpeg -an -i ' + videoPath + ' -filter:v "setpts='+newMult+'*PTS" '+ videoOutputPath

# subprocess.call(c, shell=True)