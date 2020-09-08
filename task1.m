tic
%% Load images, resize
pink = imread('Collection 1/HE/6.3.bmp');
pink = imresize(pink, 0.6);
purple = imread('Collection 1/p63AMACR/6.3.bmp');
purple = imresize(purple, 0.6);

%% Showing the original fused image

imshow(imfuse(purple, pink, 'blend'));  

%% Converting & extracting matched coord
pink = single(rgb2gray(pink));      %Turning images into grayscale and
purple = single(rgb2gray(purple));  %single precision

[k1,d1] = vl_sift(pink);            %Computing SIFT keypoints and
[k2,d2] = vl_sift(purple);          %descriptors

matches = vl_ubcmatch(d1, d2);      %Finding preliminary matches of pairs
                                    %of keypoints using descriptors
pinkpoints = zeros(2, length(matches));
purplepoints = zeros(2, length(matches));

for i =1:length(matches)               %Extracting coordinates for the
pinkpoints(1,i) = k1(1, matches(1,i)); %pink and purple matched points
pinkpoints(2,i) = k1(2, matches(1,i));
purplepoints(1,i) = k2(1, matches(2,i));
purplepoints(2,i) = k2(2, matches(2,i));
end


%% RANSAC Implementation

for i = 1:10
r1 = randi(length(matches));            %Randomly selecting points for r1
r2 = randi(length(matches));            %and for r2 from matches

[R,t] = rigreg([pinkpoints(:,r1) pinkpoints(:,r2)], [purplepoints(:,r1) purplepoints(:,r2)]);
%Using the function rigreg to calculate R and t
Transform = @(x) R*x + t;                           %Calculating the transform T
pinkpoints_warp = Transform(pinkpoints(:,1:end));   %and applying it on the pink points

distance = abs(pinkpoints_warp - purplepoints);     %Distance between points
d = zeros(1, length(distance));
for m = 1:length(distance)
    d(m)=sqrt((distance(1,m))^2 + (distance(2, m))^2);
end

inliers_count_max = 0;
inliner_max = [];
inliers_count = 0;
temp_inliers = [];
for n = 1:length(d)
    if d(n) < 50                                %Threshold for distance
        inliers_count = inliers_count + 1;
        temp_inliers = [temp_inliers n];        %Saving the index for the inliers
    end
end
    if inliers_count_max < inliers_count
        inliers_count_max = inliers_count;      %Saving the maximum value
        inliers_max = temp_inliers;             %of inliers and its index
    end
end
inliers_count_max
%% Calculating the resulting fused image

true_pinkpoints = zeros(2, length(inliers_max));
true_purplepoints = zeros(2, length(inliers_max));
for i=1:length(inliers_max)
   true_pinkpoints(:, i) = pinkpoints(:, inliers_max(i));   %Saving the true matches
   true_purplepoints(:,i) = purplepoints(:, inliers_max(i));%for the images
end


[R, t]=rigreg(true_pinkpoints, true_purplepoints);      %Calculating the
                                                        %true R and t
T = [R, t ;                                             
0, 0, 1];

tform = affine2d(T');
pink_warp = imwarp(pink, tform, 'OutputView', imref2d(size(purple))); %Lastly creating the correct
                                                                      %rigid registration                                                                     
figure
imshow(imfuse(purple, pink_warp, 'blend'));              %Showing the resulting fused image


                                                                      
translation_magnitude = norm(t)                          %Magnitude of translation vector
rotation_angle = acos(R(1,1))*180/pi                     %Rotation angle

toc
