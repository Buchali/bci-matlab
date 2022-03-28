%% Preview
% This file:
%           1. loads EEG data
%           2. converts EEG data to stft images
%           3. rescale stft images to (64 * 64)
%           3. stores the stft images as imageDatastore.

%% Add dataset dir to Matlab path
current_dir = pwd;
src_dir = fileparts(pwd);
data_dir = fullfile(src_dir, 'data/Graz_dataset');
addpath(data_dir);

%% Load MI signals
file_name = 'BCIcomp2dataset3';
load(file_name, 'X', 'y');
fs = 128;


%% stft

% temporal filter: [3.25s to 6.25s]
X = X(floor(3.25 * fs) + 1 : floor(6.25 * fs), :, :);

for i = 1 : size(X, 3)
        [tmp, f] = stft(X(:, :, i), fs, 'Window', hann(32, 'periodic'), ...
            'OverlapLength', 25, 'FFTlength', 128);
        ind_freq = (f>5 & f<31);
        tmp = abs(tmp(ind_freq, :, :));
        img(:, :, i) = reshape(permute(tmp, [1, 3, 2]), [], size(tmp, 2)); % 3D to 2D convert
end

%% rescale
img = imresize(img, [64, 64]);

%% sort dataset based on their class: y = 0 or y = 1
[y, ind] = sort(y);
ind_split = find(y == min(y), 1, 'last');
img = img(:, :, ind);

%% save image data
image_data_dir = fullfile(data_dir, 'image_data');

% class 0
% create a dir for class 0 images if not exist
class_0_dir = fullfile(image_data_dir, '0');
if not(isfolder(class_0_dir))
    mkdir(class_0_dir);
end
cd(class_0_dir);
for i = 1 : ind_split
    imwrite(img(:, :, i), [num2str(i), '.png'])
end

% class 1
% create a dir for class 1 images if not exist
class_1_dir = fullfile(image_data_dir, '1');
if not(isfolder(class_1_dir))
    mkdir(class_1_dir);
end
cd(class_1_dir);
for i = ind_split + 1 : size(img,3)
    imwrite(img(:, :, i), [num2str(i), '.png'])
end

cd(current_dir)
disp('image data are created successfully!')
