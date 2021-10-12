# PROBLEM 2
import cv2
import numpy as np
from matplotlib import pyplot as plt

window_size = 2
threshold = 750

img = cv2.imread('exam1_pics/prob2.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)

offset = int(window_size/2)
y_range = img.shape[0] - offset
x_range = img.shape[1] - offset

#Calculate the derivatives
Iy, Ix = np.gradient(img)
Ixx = Ix*Ix
Ixy = Ix*Iy
Iyy = Iy*Iy

# Blur didn't seem to make a difference for this picture
# Ixx = cv2.GaussianBlur(Ixx, (3,3), 1)
# Ixy = cv2.GaussianBlur(Ixy, (3,3), 1)
# Iyy = cv2.GaussianBlur(Iyy, (3,3), 1)

all =[] # stores all f values

for y in range(offset, y_range):
    for x in range(offset, x_range):
        start_y = y - offset
        end_y = y + offset + 1
        start_x = x - offset
        end_x = x + offset + 1

        windowIxx = Ixx[start_y : end_y, start_x : end_x]
        windowIxy = Ixy[start_y : end_y, start_x : end_x]
        windowIyy = Iyy[start_y : end_y, start_x : end_x]

        Sxx = windowIxx.sum()
        Sxy = windowIxy.sum()
        Syy = windowIyy.sum()

        # H = [[Sxx, Sxy],
        #      [Sxy, Syy]]

        det = (Sxx * Syy) - (Sxy*Sxy)
        tr = Sxx + Syy

        f = det / (tr + .01) # .01 to correct for divide by 0 error
        if f > threshold:
            all.append(f)
        else:
            all.append(0)


all = np.array(all).reshape(img.shape[0]-2, img.shape[1]-2) # reformat f values to image array

# non-maxima suppression for 3x3 window
for y in range(offset, y_range-1):
    for x in range(offset, x_range-1):
       if all[y][x] > threshold:
          if all[y][x] > all[y-1][x-1] and all[y][x] > all[y-1][x] and all[y][x] > all[y-1][x+1] and all[y][x] > all[y][x-1] and all[y][x] > all[y][x+1] and all[y][x] > all[y+1][x-1] and all[y][x] > all[y+1][x] and all[y+1][x+1]:
             cv2.circle(corners,(x, y), 2, (255,0,0))

plt.imshow(corners)
plt.xticks([]), plt.yticks([])
plt.show()
