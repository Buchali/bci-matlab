function signal2image(X, y, dataset_dir)
% this function converts EEG signal to image data
disp('imageDatastore creation...')
%% params
fs = 128;
temporal_range = [3.25, 6.25];
electrodes = [1, 3]; % Cz is not necessary

%% clean data
% temp/spatial filter: [3.25s to 6.25s] + remove Cz data

X = X(floor(temporal_range(1) * fs) + 1 : floor(temporal_range(2) * fs), electrodes, :);

%% sort dataset based on their class: y = 0 or y = 1
[y, ind] = sort(y);
ind_split = find(y == min(y), 1, 'last');
X = X(:, :, ind);

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
    imwrite(X(:, :, i), [num2str(i), '.png'])
end

% class 1
% create a dir for class 1 images if not exist
class_1_dir = fullfile(image_data_dir, '1');
if not(isfolder(class_1_dir))
    mkdir(class_1_dir);
end
cd(class_1_dir);
for i = ind_split + 1 : size(X,3)
    imwrite(X(:, :, i), [num2str(i), '.png'])
end

disp('imageDatastore created successfully!')
