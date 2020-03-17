import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dev functions start

# count the number of frames in a video
def countFrames(video):
    total = 0

    while True:
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        total += 1

    return total

# write frames
def makeFramesForDev(videoPath):
    video = cv2.VideoCapture(videoPath)
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite('frame' + str(i) + '.jpg', frame)
        i += 1

    video.release()
    cv2.destroyAllWindows()

# Dev functions end here


class Vertices:
    def __init__(self, imageShape, lowerLeft, lowerRight, topLeft, topRight):
        self.imageShape = imageShape
        self.lowerLeft = lowerLeft
        self.lowerRight = lowerRight
        self.topLeft = topLeft
        self.topRight = topRight


def drawLines(img, lines, color=None, thickness=2):
    if color is None:
        color = [0, 0, 255]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def houghLines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    drawLines(line_img, lines)
    return line_img

def weightedImage(img, initialImage, a=0.8, ß=1., λ=0.):
    return cv2.addWeighted(initialImage, a, img, ß, λ)

def regionOfInterest(image, vertices):
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channelCount = image.shape[2]
        ignoreMaskColor = (255,) * channelCount
    else:
        ignoreMaskColor = 255

    cv2.fillPoly(mask, vertices, ignoreMaskColor)

    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage


def getResizeImage(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def getCannyEdges(gauss_gray):
    lowThreshold = 60
    highThreshold = 60*3
    return cv2.Canny(gauss_gray, lowThreshold, highThreshold)

def getVertices(image):
    imageShape = image.shape
    lowerLeft = [0, imageShape[0]]
    lowerRight = [imageShape[1], imageShape[0]]
    topLeft = [0, 0]
    topRight = [imageShape[1], 0]

    v = Vertices(imageShape, lowerLeft, lowerRight, topLeft, topRight)
    return [np.array([v.lowerLeft, v.topLeft, v.topRight, v.lowerRight], dtype=np.int32)]

def adjustGamma(image, gamma=0.6):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

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

def removeVehicles(videoPath):

    # Get the video
    video = cv2.VideoCapture(videoPath)

    # empty array that will store each frame of the video
    framesList = []

    # Loop through the video for number of frames
    while True:
        # Get the frame
        ret, frame = video.read()

        # If frame is null then this is the end of file
        # so break out of loop
        if frame is None:
            break

        # If frame is not null
        # convert resize the frame and convert it to greyscale
        gray = getGrayScale(getResizeImage(frame, 800, 600))

        # And add it in the frames List
        framesList.append(gray)

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

    #         Then continue for every pixel in the frameList

    fix, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    return newImage

def adjustVideoLength(videoPath):
    print('test')


def main():

    # Set the video path
    videoPath = 'fast3.mp4'

    dev = False
    if dev:
        print(countFrames(cv2.VideoCapture(videoPath)))

    # removeVehicle function tries to remove any moving
    # objects from the video and returns an image
    roadImage = removeVehicles(videoPath)

    # Find white color in the filtered image
    maskWhite = cv2.inRange(roadImage, 200, 255)

    # Get canny edges of image
    cannyEdges = getCannyEdges(maskWhite)

    # ROI is the whole image
    vertices = getVertices(roadImage)
    ROIImage = regionOfInterest(cannyEdges, vertices)

    # Don't know what this shit is
    rho = 2
    theta = np.pi / 180
    threshold = 20
    min_line_len = 50
    max_line_gap = 200

    # Make colored lines
    linedImage = houghLines(ROIImage, rho, theta, threshold, min_line_len, max_line_gap)

    # Combine it with original image
    result = weightedImage(linedImage, getRGB(roadImage), a=0.8, ß=1., λ=0.5)

    # Show Images
    showImage(roadImage, 'Road')
    showImage(result, 'Result')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
