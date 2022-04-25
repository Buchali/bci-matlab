function dcgan(imds, batch_size, num_generated, gendata_dir)

% this function runs a DCGAN model on stft images of EEG signals.
disp('Loading the model...')
current_dir = pwd;

%% augmentation
image_size = [32, 32, 1];
auimds = augmentedImageDatastore(image_size, imds);
auimds.MiniBatchSize = batch_size;

%% generator layers
filterSize = [4 4];
numFilters = 32;
numLatentInputs = 100;

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    transposedConv2dLayer(filterSize,8*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,4*numFilters,'Stride',2,'Cropping',1,'Name','tconv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Cropping',1,'Name','tconv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    %     transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping',1,'Name','tconv4')
    %     batchNormalizationLayer('Name','bn4')
    %     reluLayer('Name','relu4')
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',1,'Name','tconv5')
    tanhLayer('Name','tanh')];

lgraphGenerator = layerGraph(layersGenerator);
dlnetGenerator = dlnetwork(lgraphGenerator);

%% discriminator layers
scale = 0.2;

layersDiscriminator = [
    imageInputLayer([32 32 1],'Normalization','none','Name','in')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding',1,'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding',1,'Name','conv2')
    batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding',1,'Name','conv3')
    batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding',1,'Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(2,1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

%% Training options
numEpochs = 20;
miniBatchSize = batch_size;

learnRateGenerator = 0.0002;
learnRateDiscriminator = 0.0001;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

%% Training

ZValidation = randn(1,1,numLatentInputs,16,'single');
dlZValidation = dlarray(ZValidation,'SSCB');


figure
iteration = 0;
start = tic;

disp('Model loaded!')
disp('Starting the training...')
% Loop over epochs.
for i = 1:numEpochs
    disp(['Epoch #' , num2str(i)])
    % Reset and shuffle datastore.
    reset(auimds);
    %     augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(auimds)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        data = read(auimds);
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.
        X = cat(4,data{:,1}{:});
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Normalize the images
        X = (single(X)/255)*2 - 1;
        
        % Convert mini-batch of data to dlarray specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');

        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRateDiscriminator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRateGenerator, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every 100 iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,20) == 0 || iteration == 1
            
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            % Rescale the images in the range [0 1] and display the images.
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            imagesc(I)
            
            % Update the title with training progress information.
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            drawnow
        end
    end
end

disp('Training is done!')
%% generate new images
ZNew = randn(1,1,numLatentInputs,num_generated,'single');
dlZNew = dlarray(ZNew,'SSCB');

dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

XGenerated = extractdata(squeeze(dlXGeneratedNew));
%% save image data

if not(isfolder(gendata_dir))
    mkdir(gendata_dir);
end
cd(gendata_dir);
for i = 1 : size(XGenerated,3)
    imwrite(XGenerated(:, :, i), ['g',num2str(i), '.png'])
end

cd(current_dir)
disp('generated image dataset is created successfully!')
%% gradient function
    function [gradientsGenerator, gradientsDiscriminator, stateGenerator] = ...
            modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ)
        
        % Calculate the predictions for real data with the discriminator network.
        dlYPred = forward(dlnetDiscriminator, dlX);
        
        % Calculate the predictions for generated data with the discriminator network.
        [dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
        dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);
        
        % Calculate the GAN loss
        [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated);
        
        % For each network, calculate the gradients with respect to the loss.
        gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
        gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);
        
    end

%% loss
    function [lossGenerator, lossDiscriminator] = ganLoss(dlYPred,dlYPredGenerated)
        
        % Calculate losses for the discriminator network.
        lossGenerated = -mean(log(1-sigmoid(dlYPredGenerated)));
        lossReal = -mean(log(sigmoid(dlYPred)));
        
        % Combine the losses for the discriminator network.
        lossDiscriminator = lossReal + lossGenerated;
        
        % Calculate the loss for the generator network.
        lossGenerator = -mean(log(sigmoid(dlYPredGenerated)));
        
    end
end