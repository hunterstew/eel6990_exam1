# PROBLEM 5
from stitch import Stitcher
import imutils
import cv2

imageA = cv2.imread('exam1_pics/prob5-2.jpg', cv2.COLOR_BGR2GRAY)
imageB = cv2.imread('exam1_pics/prob5-3.jpg', cv2.COLOR_BGR2GRAY)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
cv2.imwrite('exam1_answers/problem5-temp.png', result)

imageC = cv2.imread('exam1_pics/prob5-1.jpg', cv2.COLOR_BGR2GRAY)
imageD = cv2.imread('exam1_answers/problem5-temp.png', cv2.COLOR_BGR2GRAY)

(result, vis) = stitcher.stitch([imageC, imageD], showMatches=True)
cv2.imwrite('exam1_answers/problem5.png', result)

cv2.imshow('stitch', result)
cv2.waitKey(0)