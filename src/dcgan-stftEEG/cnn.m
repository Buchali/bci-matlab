function net = cnn(imds_train, num_filters, batch_size)
%% params
input_size = size(readimage(imds_train,1));

layers = [
    imageInputLayer([input_size, 1],'Normalization','zscore','Name','in')

    convolution2dLayer([input_size(1), 3],num_filters,'Stride',1,'Padding',1,'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    tanhLayer('Name','tanh1')
    
%     convolution2dLayer([1, 5],num_filters,'Stride',3,'Padding',0,'Name','conv2')
%     batchNormalizationLayer('Name','bn2')
%     tanhLayer('Name','tanh2')
    averagePooling2dLayer([1, 5], 'Stride', [1, 3], 'Name', 'pool1')

    fullyConnectedLayer(100, 'Name', 'fc1')
    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer
    classificationLayer
    ];

%% train
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs',50, ...
    'MiniBatchSize',batch_size, ...
    'InitialLearnRate',1e-3, ...
    'verbose', 0, 'verboseFrequency', 200);

%% out
disp('Training the CNN model...')
net = trainNetwork(imds_train, layers, options);
