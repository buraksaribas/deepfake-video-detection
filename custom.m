dataset = imageDatastore('frames10', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');

[training_data, validation_data, test_data] = splitEachLabel(dataset, 0.7, 0.15, 0.15, 'randomized');

input_size =[256 256 3];
resized_training_data = augmentedImageDatastore(input_size, training_data);
resized_validation_data = augmentedImageDatastore(input_size, validation_data);
resized_test_data = augmentedImageDatastore(input_size, test_data);

layers = [
    imageInputLayer([256 256 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(4,'Stride',4)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-2, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_validation_data, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(resized_training_data, layers, options);

pred = classify(net,resized_test_data);

figure
cm = confusionchart(test_data.Labels,pred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

save('custom_frames10.mat', 'net');
