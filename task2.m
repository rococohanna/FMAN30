%% Load images, resize
trf = imread('Collection 2/TRF/7TRF.tif');
trf = imresize(trf, 0.6);
he = imread('Collection 2/HE/7HE.jpg');
he = imresize(he, 0.6);

%% Showing the original fused image

imshow(imfuse(trf, he, 'blend')); 

%% Converting & extracting matched coord
trf = single(imcomplement(histeq(trf)));   %Enhancing contrast using histogram 
he = single(rgb2gray(he));                 %equalization and using the inverse

[k1,d1] = vl_sift(trf);                    %Computing SIFT keypoints and
[k2,d2] = vl_sift(he);                     %descriptors

matches = vl_ubcmatch(d1, d2);             %Finding preliminary matches of pairs
                                           %of keypoints using descriptors
trfpoints = zeros(2, length(matches));     
hepoints = zeros(2, length(matches));

for i =1:length(matches)                   %Extracting coordinates for the
trfpoints(1,i) = k1(1, matches(1,i));      %TRF and HE matched points
trfpoints(2,i) = k1(2, matches(1,i));
hepoints(1,i) = k2(1, matches(2,i));
hepoints(2,i) = k2(2, matches(2,i));
end

%% RANSAC Implementation

inliers_count_max = 0;
inliner_max = [];
for i = 1:20
r1 = randi(length(matches));               %Randomly selecting points for r1
r2 = randi(length(matches));               %and r2 from matches

[R,t,s] = rigreg_2([trfpoints(:,r1) trfpoints(:,r2)], [hepoints(:,r1) hepoints(:,r2)]);
%Using the function rigreg to calculate R, t and s

Transform = @(x) s*R*x + t;             %Calculating the transform T and applying it on
trfpoints_warp = Transform(trfpoints(:,1:end));     %the TRF points

distance = abs(trfpoints_warp - hepoints);
d = zeros(1, length(distance));
for m = 1:length(distance)
    d(m)=sqrt((distance(1,m))^2 + (distance(2, m))^2);
end

inliers_count = 0;
temp_inliers = [];
for n = 1:length(d)
    if d(n) < 50                           %Threshold for distance
        inliers_count = inliers_count + 1;
        temp_inliers = [temp_inliers n];   %Saving the index for the inliers
    end
end
    if inliers_count_max < inliers_count
        inliers_count_max=inliers_count;   %Saving the maximum value
        inliers_max = temp_inliers;        %of inliers and its index
    end
end
inliers_count_max


%% Calculating the resulting fused image

true_trfpoints = zeros(2, length(inliers_max));
true_hepoints = zeros(2, length(inliers_max));
for i=1:length(inliers_max)
   true_trfpoints(:, i) = trfpoints(:, inliers_max(i)); %Saving the true matches
   true_hepoints(:,i) = hepoints(:, inliers_max(i));    %for the images
end

[R, t, s]=rigreg_2(true_trfpoints, true_hepoints);  %Calculating the true R, t and s

T = [s*R, t ;
0, 0, 1];

tform = affine2d(T');
trf_warp = imwarp(trf, tform, 'OutputView', imref2d(size(he))); %Lastly creating the correct
figure                                                          %rigid registration
imshow(imfuse(he, trf_warp, 'blend'));              %Showing the resulting fused image

translation_magnitude = norm(t)                     %Magnitude of translation vector
rotation_angle = acos(R(1,1))*180/pi                %Rotation angle
scale_factor = s                                    %Scaling factor        