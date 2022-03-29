function model(dataset_dir, num_folds, batch_ratio)

disp('Loading the model...')
%% load dateset as Datastore
imgs = imageDatastore(dataset_dir, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');

%% params
k = num_folds; % folds
input_size = [size(readimage(imgs,1)), 1]; % [384, 2, 1]

fold = floor(size(imgs.Labels,1) / k); % length of folds


batch_size = round((size(imgs.Labels,1) - fold) * batch_ratio);

%% shuffle data
rng(9)
shuffle_ind = randperm(size(imgs.Labels, 1));
imgs = subset(imgs, shuffle_ind);

%% layers
activation = leakyReluLayer;

layers = [
    imageInputLayer(input_size, 'Name', 'input') % out: [384, 2, 1]
    
    convolution2dLayer([10, 1], 15, 'Name', 'conv1',...
        'stride', [1, 1], 'WeightsInitializer', 'narrow-normal') % out: [375, 2, 15]
    activation
%     dropoutLayer(0.3)
    
    convolution2dLayer([1, 2], 15, 'Name', 'conv2',...
        'stride', [1, 1], 'WeightsInitializer', 'narrow-normal') % out: [375, 1, 15]
    batchNormalizationLayer
    activation
    
    averagePooling2dLayer([3, 1], ...
        'Name', 'pool1', 'stride', [3, 1]) % out: [125, 1, 15]
    
    convolution2dLayer([10, 1], 30, 'Name', 'conv3',...
        'stride', [1, 1], 'WeightsInitializer', 'narrow-normal') % out: [116, 1, 30]
    activation
%     dropoutLayer(0.2)
    
    averagePooling2dLayer([5, 1], ...
        'Name', 'pool2', 'stride', [3, 1]) % out: [37, 1, 30]
        
    convolution2dLayer([11, 1], 60, 'Name', 'conv4',...
        'stride', [1, 1], 'WeightsInitializer', 'narrow-normal') % out: [27, 1, 60]
    batchNormalizationLayer
    activation
%     dropoutLayer(0.1)
    
    averagePooling2dLayer([7, 1], ...
        'Name', 'pool3', 'stride', [4, 1]) % out: [6, 1, 60]
    
    fullyConnectedLayer(2, 'Name', 'fc', 'WeightL2Factor', 0.5)
    
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];

%% training options
options = trainingOptions('adam', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.8, ...
    'LearnRateDropPeriod', 10, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', batch_size, ...
    'InitialLearnRate', 1e-3, ...
    'verbose', false, ...
    'verboseFrequency', 50, ...
    'Plots', 'none');

%% train and test
disp('Train and Test process begins...')
disp('----------------------------')
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

disp('----------------------------')
disp(['Mean accuracy: ', num2str(accuracy_total)])
disp('----------------------------')
