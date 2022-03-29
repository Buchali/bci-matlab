clc; close all;
%% Params
flag_create_images = false;

%% Add dataset dir to Matlab path
current_dir = pwd;
src_dir = fileparts(pwd);
data_dir = fullfile(src_dir, 'data/Graz_dataset');
addpath(data_dir);

dataset_dir = fullfile(data_dir, 'raw_image_data');

%% signal to image conversion

% check if the image dataset already exists.
if (~isfolder(dataset_dir) || flag_create_images)
% Load MI data
file_name = 'BCIcomp2dataset3';
load(file_name, 'X', 'y');

signal2image(X, y, dataset_dir)
else
    disp('Image dataset already exists')
end

cd(current_dir)
%% model
num_folds = 10;
batch_ratio = 0.25;

% train and test model
model(dataset_dir, num_folds, batch_ratio)
disp('Done')
