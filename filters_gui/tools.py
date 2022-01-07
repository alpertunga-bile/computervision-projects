import cv2
from scipy.interpolate import UnivariateSpline
import numpy as np

#----------------------------------------------------------------------------------------------

def dodge(image, mask):
    return cv2.divide(image, 255 - mask, scale = 256)

def convert2pencilSketch(reel_image):
    grayImage = cv2.cvtColor(reel_image, cv2.COLOR_RGB2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (21, 21), 0, 0)
    sketchImage = dodge(grayImage, blurredImage)
    return cv2.cvtColor(sketchImage, cv2.COLOR_GRAY2RGB)

def convert2pencilSketchWCanvas(reel_image, isCanvas = None):
    grayImage = cv2.cvtColor(reel_image, cv2.COLOR_RGB2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (21, 21), 0, 0)
    sketchImage = dodge(grayImage, blurredImage)
    if isCanvas is not None:
        sketchImage = cv2.multiply(sketchImage, isCanvas, scale = 1 / 256)
    return cv2.cvtColor(sketchImage, cv2.COLOR_GRAY2RGB)

#----------------------------------------------------------------------------------------------

def spline2LookupTable(splineBreaks: list, breakValues: list):
    spl = UnivariateSpline(splineBreaks, breakValues)
    return spl(range(256))

def applyRGBFilters(reelImage, *, redFilter = None, greenFilter = None, blueFilter = None):
    red, green, blue = cv2.split(reelImage)
    
    if redFilter is not None:
        red = cv2.LUT(red, redFilter).astype(np.uint8)
    if greenFilter is not None:
        green = cv2.LUT(green, greenFilter).astype(np.uint8)
    if blueFilter is not None:
        blue = cv2.LUT(blue, blueFilter).astype(np.uint8)

    return cv2.merge((red, green, blue))

def applyHUEFilter(reelImage, hueFilter):
    hue, saturation, value = cv2.split(cv2.cvtColor(reelImage, cv2.COLOR_RGB2HSV))

    saturation = cv2.LUT(saturation, hueFilter).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2RGB)

#----------------------------------------------------------------------------------------------

def cartoonize(reelImage, *, numPyrDowns = 2, numBilaterals = 7):
    # Apply Bilateral Filter
    downsampledImg = reelImage
    for _ in range(numPyrDowns):
        downsampledImg = cv2.pyrDown(downsampledImg)

    for _ in range(numBilaterals):
        filteredDownsampledImg = cv2.bilateralFilter(downsampledImg, 9, 9, 7)
    
    filteredNormalImg = filteredDownsampledImg
    for _ in range(numPyrDowns):
        filteredNormalImg = cv2.pyrUp(filteredNormalImg)
    
    if filteredNormalImg.shape != reelImage.shape:
        filteredNormalImg = cv2.resize(filteredNormalImg, reelImage[:2])

    # Convert original image into grayscale
    grayImg = cv2.cvtColor(reelImage, cv2.COLOR_RGB2GRAY)

    # Apply median blur to reduce image noise
    blurImg = cv2.medianBlur(grayImg, 7)

    # Use adaptive threshold to detect edges
    grayEdges = cv2.adaptiveThreshold(blurImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 2)

    rgbEdges = cv2.cvtColor(grayEdges, cv2.COLOR_GRAY2RGB)

    # combine color image and edge mask
    return cv2.bitwise_and(filteredNormalImg, rgbEdges)