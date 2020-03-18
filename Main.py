import cv2
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Globals
originalVideoPath = 'video7.mp4'
tempVideoPath = 'temp.mp4'

def main():

    # Set the video paths
    global originalVideoPath
    global tempVideoPath

    # Adjust video length to make it 10 seconds video
    adjustVideoLength(originalVideoPath, 10)

    dev = True
    if dev:
        print(str(getTotalFramesOfVideo(tempVideoPath)) + ' total frames')
        print(str(getDurationOfVideo(tempVideoPath)) + ' seconds')

    # removeVehicle function tries to remove any moving
    # objects from the video and returns an image
    roadImage = removeVehicles(tempVideoPath)

    # Find white color in the filtered image
    maskWhite = cv2.inRange(roadImage, 200, 255)

    # Binary conversion of image
    (thresh, blackAndWhiteImage) = cv2.threshold(maskWhite, 127, 255, cv2.THRESH_BINARY)
    x, y = blackAndWhiteImage.shape

    # Get road lined from image
    linedImage = detectRoadLines(blackAndWhiteImage)

    # Combine the images with detected lines and original image
    result = weightedImage(linedImage, getRGB(roadImage), a=0.3, ß=1., λ=0.5)

    # Combine the original video and lined image obtained
    combineRoadAndVideo(linedImage)

    # Show Images
    # showImage(roadImage, 'Road')
    # showImage(linedImage, 'Lined')
    # showImage(result, 'Result')

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def combineRoadAndVideo(image):
    global originalVideoPath
    # Read the original video
    video = cv2.VideoCapture(originalVideoPath)

    while video.isOpened():
        ret, frame = video.read()
        # Resize frame
        resizedFrame = getResizeImage(frame, 800, 600)
        if ret:
            # Combine both images
            combinedImage = weightedImage(image, resizedFrame, a=0.6, ß=1., λ=0.8)
            cv2.imshow('Frame', getResizeImage(combinedImage, 800, 600))
            cv2.waitKey(0)
        else:
            break
    video.release()

    cv2.destroyAllWindows()

# Make any video of desired length by making it
# slower or faster
def adjustVideoLength(videoPath, newDuration):
    global tempVideoPath

    # Get FPS of the original video
    FPS = getFPS(videoPath)

    # Get total number of frames in original video
    totalFrames = getTotalFramesOfVideo(videoPath)

    # Divide both to get the duration of video in seconds
    originalDuration = totalFrames / FPS

    # Now calculate the multiplier to make the video of 10 seconds
    multiplier = str(newDuration / originalDuration)

    # Using ffmpeg to make the video faster or slower
    c = 'ffmpeg -an -i ' + videoPath + ' -filter:v "setpts=' \
        + multiplier + '*PTS" ' + tempVideoPath
    subprocess.call(c, shell=True)

    print('video converted')


# Remove moving vehicles from a video
def removeVehicles(videoPath):

    # Get the video
    video = cv2.VideoCapture(videoPath)

    # empty array that will store each frame of the video
    framesList = []

    # Loop through the video for number of frames
    skipper = 0
    while True:
        skipper += 1
        # Get the frame
        ret, frame = video.read()

        # If frame is null then this is the end of file
        # so break out of loop
        if frame is None:
            break

        if skipper % 2 is not 0:
            continue

        # If frame is not null
        # convert resize the frame and convert it to greyscale
        gray = getGrayScale(getResizeImage(frame, 800, 600))

        # And add it in the frames List
        framesList.append(gray)

    # Release video
    video.release()

    # Get the number of rows and cols of the first frame
    # Only getting of first frame because all are same
    x, y = framesList[0].shape

    # Make a copy of the image this is the image in which
    # result will be stored
    newImage = np.copy(framesList[0])

    # Iterate over the frames for number of rows and cols
    for row in range(x - 1):
        for col in range(y - 1):

            argsList = []

            # For every pixel of the same location in frameList
            # append it to argsList
            # e.g the below loop after running for the number of total frames
            # will append in argsList
            # frame0(0,0) , frame1(0,0), frame2(0,0) and so on
            for i in range(len(framesList)):
                argsList.append(framesList[i][row][col])

            # Take the mean of all the items in the argsList
            mean = np.mean([argsList]).astype(int)

            # Assign the mean value to the resultImage at the same location
            newImage[row][col] = mean

    # Then continue for every pixel in the frameList

    fix, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    return newImage


def detectRoadLines(blackAndWhiteImage):
    x, y = blackAndWhiteImage.shape

    # Make copy of the image (result image)
    linedImage = np.copy(getRGB(blackAndWhiteImage))

    # Iterate over the image
    for row in range(x - 1):
        for col in range(y - 1):
            # If value above 127 make it red
            if blackAndWhiteImage[row][col] > 127:
                linedImage[row][col][0] = 0
                linedImage[row][col][1] = 255
                linedImage[row][col][2] = 0
            # Else make it black
            else:
                linedImage[row][col][0] = 0
                linedImage[row][col][1] = 0
                linedImage[row][col][2] = 0
    return linedImage


# Get FPS of video
def getFPS(videoPath):
    fps = cv2.VideoCapture(videoPath).get(cv2.CAP_PROP_FPS)
    return fps


# Count the number of frames in a video
def getTotalFramesOfVideo(videoPath):
    totalFrames = cv2.VideoCapture(videoPath).get(cv2.CAP_PROP_FRAME_COUNT)
    return totalFrames


# Get the total duration of a video in seconds
def getDurationOfVideo(videoPath):
    # duration = totalFrames in video / frames per seconds of video
    duration = getTotalFramesOfVideo(videoPath) / getFPS(videoPath)
    return duration


def drawLines(img, lines, color=None, thickness=2):
    if color is None:
        color = [0, 0, 255]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# Combine two images
def weightedImage(img, initialImage, a=0.8, ß=1., λ=0.):
    return cv2.addWeighted(initialImage, a, img, ß, λ)


# Resize an image
def getResizeImage(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


# Find Edges in an image
def getCannyEdges(image):
    lowThreshold = 60
    highThreshold = 60*3
    return cv2.Canny(image, lowThreshold, highThreshold)


# Adjust gamma of an image
def adjustGamma(image, gamma=0.6):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# adjust saturation of an image
def adjustSaturation(image):
    # contrast
    alpha = 1.2
    # brightness
    beta = 0.2
    new_image = np.zeros(image.shape, image.dtype)
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image


# Converts RGB to Greyscale image
def getGrayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Converts Greyscale image to RGB image
def getRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


# Apply Gaussian blur on given image
def getGaussianBlur(maskWhite):
    # You can set the mask size here
    # Odd mask size is better
    maskSize = 3
    return cv2.GaussianBlur(maskWhite, (maskSize, maskSize), 0)


# Function to show image with title on screen
def showImage(image, title):
    cv2.imshow(title, getResizeImage(image, 800, 600))


class Vertices:
    def __init__(self, imageShape, lowerLeft, lowerRight, topLeft, topRight):
        self.imageShape = imageShape
        self.lowerLeft = lowerLeft
        self.lowerRight = lowerRight
        self.topLeft = topLeft
        self.topRight = topRight


if __name__ == '__main__':
    main()
