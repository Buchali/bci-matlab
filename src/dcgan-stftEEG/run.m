clc; close all;
%% params
flag_preprocess = true; % if you want to create image dataset set this to 'true'.

%% add dataset dir to Matlab path
current_dir = pwd;
src_dir = fileparts(pwd);
data_dir = fullfile(src_dir, 'data/Graz_dataset');
addpath(data_dir);

dataset_dir = fullfile(data_dir, 'stft_image_data');

%% signal to image conversion

% check if the image dataset already exists.
if (~isfolder(dataset_dir) || flag_preprocess)
    disp('Creating the stft image dataset...')
    % Load MI data
    file_name = 'BCIcomp2dataset3';
    load(file_name, 'X', 'y');
    
    preprocess(X, y, dataset_dir)
else
    disp('Image dataset already exists')
end

cd(current_dir)
%% model
num_folds = 10;
batch_size = 28;

% train and test model
model(dataset_dir, batch_size)
disp('Done')