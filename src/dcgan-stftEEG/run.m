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
disp('-------------------------------')
%% DCGAN Data Augmentation
% params
k_fold = 10;
cnn_num_filters = 30;
cnn_batch = 28;
accuracy = zeros(k_fold, 1);
gan_batch_size = 28;
num_generated = 56;

% load imds
gendata_dir = fullfile(data_dir, 'generated_data');

imds0 = imageDatastore(fullfile(dataset_dir,'0'), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %

imds1 = imageDatastore(fullfile(dataset_dir,'1'), ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %

num_data = size(imds0.Labels, 1);
c = cvpartition(num_data, 'Kfold', k_fold);

% k-fold crossvalidation loop
for i = 1:k_fold
    train_ind = c.training(i);
    test_ind = c.test(i);
    
    disp('---------DCGAN------------')
    % class 0
    [imds0train, imds0test] = train_test_split(imds0, train_ind, test_ind);
    gensubset0_dir = fullfile(gendata_dir, '0');
    dcgan(imds0train, gan_batch_size, num_generated, gensubset0_dir, image_size)
    
    % class 1
    [imds1train, imds1test] = train_test_split(imds1, train_ind, test_ind);
    gensubset1_dir = fullfile(gendata_dir, '1');
    dcgan(imds1train, gan_batch_size, num_generated, gensubset1_dir, image_size)
    
    imds_gen = imageDatastore(gendata_dir, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames'); %
    
    imds_train = merge_datastore(imds0train, imds1train, imds_gen);
    imds_train = shuffle_datastore(imds_train, random_seed);
    
    imds_test = merge_datastore(imds0test, imds1test);
    imds_test = shuffle_datastore(imds_test, random_seed);
    
    %% Classification
    disp('---------CLASSIFICATION------------')
    net = cnn(imds_train, cnn_num_filters, cnn_batch);
    YPred = classify(net,imds_test);
    YTest = imds_test.Labels;
    accuracy(i) = (sum(YPred == YTest)/numel(YTest))*100;
    disp(['Fold #',num2str(i),' accuracy: %', num2str(accuracy(i))])
    disp('----------------------------')
end

disp('----------------------------')
disp(['Mean accuracy: ', num2str(mean(accuracy))])
disp('----------------------------')

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

function [imds_train, imds_test] = train_test_split(imds, train_ind, test_ind)
imds_train = imageDatastore(imds.Files(train_ind));
imds_train.Labels = imds.Labels(train_ind);
imds_test = imageDatastore(imds.Files(test_ind));
imds_test.Labels = imds.Labels(test_ind);
end
