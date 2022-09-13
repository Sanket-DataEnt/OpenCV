import numpy as np
import cv2 as cv
cap = cv.VideoCapture('test.mp4')
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# fgbg = cv.bgsegm.createBackgroundSubtractorMOG() # It is a gaussian mixture based background segmentation algorithm.
# fgbg = cv.bgsegm.BackgroundSubtractorGMG()
# fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=True) # It uses the same concept but the major advantage that it provides 
# is in terms of stability even when there is change in luminosity and better identification capability of shadows in the frames.
fgbg = cv.createBackgroundSubtractorKNN(detectShadows=True)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    fgmask = fgbg.apply(frame)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    cv.imshow('Frame', frame)
    cv.imshow('FG MASK Frame', fgmask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cap.release()
cv.destroyAllWindows()