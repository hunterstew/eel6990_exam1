# PROBLEM 1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

img = cv2.imread('exam1_pics/prob1.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convolute(kernel):
   pixels = []
   for i in range(0, img.shape[0]): #rows
       row = []
       for j in range(0, img.shape[1]): #cols
           pixel = img.item(i, j)
           row.append(pixel)
       pixels.append(row)

   pixels = np.array(pixels) #225x225
   pixels = pixels.astype(np.uint8)

   kernel_size = kernel.shape

   width = len(pixels[0])
   height = len(pixels)
   pixels = np.pad(pixels, 1, mode='constant') #pads a 0 to the edges of the picture

   window_matrix = []
   for i in range(0, height):
       for j in range(0, width):
           window_matrix.append(
               [
                   [pixels[row][col] for col in range(j, j + 3)]
                   for row in range(i, i + 3)
               ]
           )
   img_sampling = np.array(window_matrix)

   transform_mat = []
   for sample in img_sampling:
       transform_mat.append(
           np.sum(np.multiply(sample, kernel))
       )
   reshape_val = int(math.sqrt(img_sampling.shape[0])) #make 1x50625 into 225x225
   filtered_img = np.array(transform_mat).reshape(reshape_val, reshape_val)
   return filtered_img

#blur then save img
kernel = (1/9)*np.array([[1,1,1],[1,1,1],[1,1,1]])
mean_filtered_img = convolute(kernel)

cv2.imwrite('exam1_answers/problem1-1.png', mean_filtered_img)
mean_filtered_img = cv2.imread('exam1_answers/problem1-1.png');
mean_filtered_img = cv2.cvtColor(mean_filtered_img, cv2.COLOR_BGR2GRAY);

#sharpen then save img
kernel = np.array([[0,-1,0], [-1, 5,-1], [0,-1,0]])
sharpened_img = convolute(kernel);

cv2.imwrite('exam1_answers/problem1-2.png', sharpened_img)
sharpened_img = cv2.imread('exam1_answers/problem1-2.png');
sharpened_img = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2GRAY);

#display images
plt.subplot(131),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mean_filtered_img,cmap='gray'),plt.title('Mean Filter')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(sharpened_img,cmap='gray'),plt.title('Sharpened Filter')
plt.xticks([]), plt.yticks([])
plt.show()