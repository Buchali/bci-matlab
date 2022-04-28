function preprocess(X, y, dataset_dir, image_size)
% this function converts EEG signals (X) to image data using stft
% the images will be stored in dataset_dir

current_dir = pwd;
%% params
fs = 128;
temporal_range = [3.25, 6.25];
freq_range = [5, 31];
%% stft

% temporal filter: [3.25s to 6.25s]
X = X(floor(temporal_range(1) * fs) + 1 : floor(temporal_range(2) * fs), :, :);

for i = 1 : size(X, 3)
    [tmp, f] = stft(X(:, :, i), fs, 'Window', hann(32, 'periodic'), ...
        'OverlapLength', 25, 'FFTlength', 128);
    ind_freq = (f > freq_range(1) & f < freq_range(2));
    tmp = abs(tmp(ind_freq, :, :));
    img(:, :, i) = reshape(permute(tmp, [1, 3, 2]), [], size(tmp, 2)); % 3D to 2D convert
end

%% normalize and rescale
img = imresize(img, image_size);
img = normalize2d(img);

%% sort dataset based on their class: y = 0 or y = 1
[y, ind] = sort(y);
ind_split = find(y == min(y), 1, 'last');
img = img(:, :, ind);

%% save image data
image_data_dir = dataset_dir;

% class 0
% create a dir for class 0 images if not exist
class_0_dir = fullfile(image_data_dir, '0');
if not(isfolder(class_0_dir))
    mkdir(class_0_dir);
end
cd(class_0_dir);
for i = 1 : ind_split
    imwrite(img(:, :, i), ['0' num2str(i), '.png'])
end

% class 1
% create a dir for class 1 images if not exist
class_1_dir = fullfile(image_data_dir, '1');
if not(isfolder(class_1_dir))
    mkdir(class_1_dir);
end
cd(class_1_dir);
for i =  1 : size(img,3) - ind_split
    imwrite(img(:, :, i+ind_split), ['1', num2str(i), '.png'])
end

cd(current_dir)
disp('image dataset is created successfully!')

function B = normalize2d(A)

A_2D(:, :, 1) = reshape(A, [size(A,1) * size(A,2) , size(A,3)]);
A_mean = reshape(mean(A_2D), [1, size(mean(A_2D))]);
A_std = reshape(std(A_2D), [1, size(std(A_2D))]);

B = (A - A_mean) ./ A_std;
B = fillmissing(B, 'constant', 0); % change NAN to 0
