
%% params
input_size = size(readimage(imds,1));
numFilters = 30;
scale = 0.2;
batch_size = 28;

layers = [
    imageInputLayer([input_size, 1],'Normalization','none','Name','in')
    convolution2dLayer([input_size(1), 3],numFilters,'Stride',1,'Padding',1,'Name','conv1')
    batchNormalizationLayer('Name','bn1')
    tanhLayer('Name','tanh1')
    
    convolution2dLayer([1, 5],2*numFilters,'Stride',3,'Padding',0,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    tanhLayer('Name','tanh2')
%     averagePooling2dLayer([1, 5], 'Stride', [1, 3], 'Name', 'pool1')

    fullyConnectedLayer(100, 'Name', 'fc1')
    fullyConnectedLayer(2, 'Name', 'fc1')
    softmaxLayer
    classificationLayer
    ];

%% train
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.8, ...
    'LearnRateDropPeriod',10, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',batch_size, ...
    'InitialLearnRate',1e-3, ...
    'verbose', 1, 'verboseFrequency', 100);

%% test
net = trainNetwork(imds, layers, options);
YPred = classify(net,imds_test);
YTest = imds_test.Labels;
accuracy = (sum(YPred == YTest)/numel(YTest))*100;
disp(['Accuracy is: %', accuracy])
