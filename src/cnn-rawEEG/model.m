
%% load dateset as Datastore
current_dir = pwd;
src_dir = fileparts(pwd);
dataset_dir = fullfile(src_dir, 'data/Graz_dataset/image_data');
imgs = imageDatastore(dataset_dir, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

%% shuffle data
rng(9)
shuffle_ind = randperm(size(imgs.Labels,1));
imgs = subset(imgs,shuffle_ind);

%%
k = 10; % folds
input_size = [size(readimage(imgs,1)), 1]; % [384, 2, 1]

fold = floor(size(imgs.Labels,1) / k); %length of folds
num_batch = round((size(imgs.Labels,1) - fold) /4);

%% Layers
layers = [
    imageInputLayer(input_size, 'Name', 'input') % out: [384, 2, 1]
    
    convolution2dLayer([10, 1], 25, 'Name', 'conv1',...
        'stride', [1, 1]) % out: [375, 2, 25]
    leakyReluLayer('Name', 'activ1')
    % dropoutLayer
    
    convolution2dLayer([1, 2], 25, 'Name', 'conv2',...
        'stride', [1, 1]) % out: [375, 1, 25]
    batchNormalizationLayer
    leakyReluLayer('Name', 'activ2')
    
    maxPooling2dLayer([3, 1], ...
        'Name', 'pool1', 'stride', [3, 1]) % out: [125, 1, 25]
    
    convolution2dLayer([12, 1], 50, 'Name', 'conv3',...
        'stride', [1, 1]) % out: [114, 1, 50]
    leakyReluLayer('Name', 'activ3')
    % dropoutLayer
    
    maxPooling2dLayer([3, 1], ...
        'Name', 'pool2', 'stride', [3, 1]) % out: [38, 1, 50]
        
    convolution2dLayer([12, 1], 100, 'Name', 'conv4',...
        'stride', [1, 1]) % out: [27, 1, 100]
    batchNormalizationLayer
    leakyReluLayer('Name', 'activ4')
    % dropoutLayer
    
    maxPooling2dLayer([3, 1], ...
        'Name', 'pool3', 'stride', [3, 1]) % out: [9, 1, 100]
    
    fullyConnectedLayer(2, 'Name', 'fc') % out: 2
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

%% training options
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs',40, ...
    'MiniBatchSize',num_batch, ...
    'InitialLearnRate',1e-3, ...
    'verbose', 0, 'verboseFrequency', 200);

%% train and test
accuracy = zeros(k, 1);
for j = 1:k
    indtest= (j-1) * fold + 1 : j * fold; % test indices
    indtrain= 1:size(imgs.Labels,1); 
    indtrain(indtest) = []; % train indices

    imgs_train = subset(imgs,indtrain);
    imgs_test = subset(imgs,indtest);

    net = trainNetwork(imgs_train, layers, options);
    
    %test
    YPred = classify(net, imgs_test);
    YTest = imgs_test.Labels;
    accuracy(j) = (sum(YPred == YTest) / numel(YTest)) * 100;
    disp(['Fold #', num2str(j), ' accuracy: ', num2str(accuracy(j))])

end
accuracy_total = mean(accuracy);
disp(['Total accuracy: ', num2str(accuracy_total)])
