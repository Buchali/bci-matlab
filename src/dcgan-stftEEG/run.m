clc; close all;
%% params
apply_preprocess = true; % if 'true', creates image dataset no matter it exists or not.
random_seed = 9;
%% add dataset dir to Matlab path
current_dir = pwd;
src_dir = fileparts(pwd);
data_dir = fullfile(src_dir, 'data/Graz_dataset');
addpath(data_dir);
dataset_dir = fullfile(data_dir, 'stft_image_data');
gendata_dir = fullfile(data_dir, 'generated_data');
%% signal to image conversion

% check if the image dataset already exists.
if (~isfolder(dataset_dir) || apply_preprocess)
    disp('Creating the stft image dataset...')
    % Load MI data
    file_name = 'BCIcomp2dataset3';
    load(file_name, 'X', 'y');
    
    preprocess(X, y, dataset_dir)
else
    disp('Image dataset already exists')
end

%% DCGAN Data Augmentation
batch_size = 28;
num_generated = 28;

imds = imageDatastore(dataset_dir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %

% class 0
[imds0train, imds0test] = splitEachLabel(imds, 0.9, 0.1, 'include','0');
gensubset0_dir = fullfile(gendata_dir, '0');
dcgan(imds0train, batch_size, num_generated, gensubset0_dir)

% class 1
[imds1train, imds1test] = splitEachLabel(imds, 0.9, 0.1, 'include','1');
gensubset1_dir = fullfile(gendata_dir, '1');
dcgan(imds1train, batch_size, num_generated, gensubset1_dir)

imds_gen = imageDatastore(gendata_dir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %
disp('Done')

imds_train = imageDatastore(cat(1, imds0train.Files, imds1train.Files, imds_gen.Files));
imds_train.Labels = cat(1, imds0train.Labels, imds1train.Labels, imds_gen.Labels);
imds_train = shuffle_ds(imds_train, random_seed);

imds_test = imageDatastore(cat(1, imds0test.Files, imds1test.Files));
imds_test.Labels = cat(1, imds0test.Labels, imds1test.Labels);
imds_test = shuffle_ds(imds_test, random_seed);

function imds_shuffled = shuffle_ds(imds, randseed)
rng(randseed)
shuffle_ind = randperm(size(imds.Labels, 1));
imds_shuffled = subset(imds, shuffle_ind);
end