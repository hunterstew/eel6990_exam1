imageFile = 'prob3-1.jpg';
inputImage = imread(imageFile);
points = detectSURFFeatures(inputImage);
imshow(inputImage);

