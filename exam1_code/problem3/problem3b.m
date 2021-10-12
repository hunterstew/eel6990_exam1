img1 = imread('prob3-1.jpg');
img2 = imread('prob3-2.jpg');

img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

points1 = detectSURFFeatures(img1);
points2 = detectSURFFeatures(img2);

[f1,vpts1] = extractFeatures(img1,points1);
[f2,vpts2] = extractFeatures(img2,points2);

indexPairs = matchFeatures(f1,f2) ;
matchedPoints1 = vpts1(indexPairs(:,1));
matchedPoints2 = vpts2(indexPairs(:,2));
disp(matchedPoints1) %26

figure; showMatchedFeatures(img1,img2,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');