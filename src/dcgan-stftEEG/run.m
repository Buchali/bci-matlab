clc; close all;
%% params
apply_preprocess = true; % if 'true', creates image dataset no matter it exists or not.
random_seed = 9;
image_size = [32, 32];

%% add dataset dir to Matlab path
current_dir = pwd;
src_dir = fileparts(pwd);
data_dir = fullfile(src_dir, 'data/Graz_dataset');
addpath(data_dir);
dataset_dir = fullfile(data_dir, 'stft_image_data');
%% stft

% check if the image dataset already exists.
if (~isfolder(dataset_dir) || apply_preprocess)
    disp('Creating the stft image dataset...')
    % Load MI data
    file_name = 'BCIcomp2dataset3';
    load(file_name, 'X', 'y');
    
    preprocess(X, y, dataset_dir, image_size)
else
    disp('Image dataset already exists')
end

%% DCGAN Data Augmentation
gendata_dir = fullfile(data_dir, 'generated_data');

batch_size = 28;
num_generated = 28;

imds = imageDatastore(dataset_dir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %

% class 0
[imds0train, imds0test] = splitEachLabel(imds, 0.9, 0.1, 'include','0');
gensubset0_dir = fullfile(gendata_dir, '0');
dcgan(imds0train, batch_size, num_generated, gensubset0_dir, image_size)

% class 1
[imds1train, imds1test] = splitEachLabel(imds, 0.9, 0.1, 'include','1');
gensubset1_dir = fullfile(gendata_dir, '1');
dcgan(imds1train, batch_size, num_generated, gensubset1_dir, image_size)

imds_gen = imageDatastore(gendata_dir, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %
disp('Done')

imds_train = merge_datastore(imds0train, imds1train, imds_gen);
imds_train = shuffle_datastore(imds_train, random_seed);

imds_test = merge_datastore(imds0test, imds1test);
imds_test = shuffle_datastore(imds_test, random_seed);

%% Classification
num_filters = 30;
batch = 28;
net = cnn(imds_train, num_filters, batch);
YPred = classify(net,imds_test);
YTest = imds_test.Labels;
accuracy = (sum(YPred == YTest)/numel(YTest))*100;
disp(['Accuracy is: %', num2str(accuracy)])


function imds_shuffled = shuffle_datastore(imds, randseed)
rng(randseed)
shuffle_ind = randperm(size(imds.Labels, 1));
imds_shuffled = subset(imds, shuffle_ind);
end

function imds = merge_datastore(varargin)
narginchk(2, inf);

files = {};
labels = [];
for i = 1:nargin
    files = cat(1, files, varargin{i}.Files);
    labels = cat(1, labels, varargin{i}.Labels);
end
imds = imageDatastore(files);
imds.Labels = labels;
end