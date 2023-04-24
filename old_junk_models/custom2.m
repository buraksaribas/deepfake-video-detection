dataset = imageDatastore('frames2', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');

[training_data, validation_data] = splitEachLabel(dataset, 0.7,'randomized');

input_size =[256 256 3];
resized_training_data = augmentedImageDatastore(input_size, training_data);
resized_validation_data = augmentedImageDatastore(input_size, validation_data);

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
        
    dropoutLayer(0.5);
    fullyConnectedLayer(16);
    leakyReluLayer(0.1);
    dropoutLayer(0.5);    
    fullyConnectedLayer(2);
    softmaxLayer;
    classificationLayer;
];

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_validation_data, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(resized_training_data, layers, options);

YPred = classify(net,resized_validation_data);

figure
cm = confusionchart(validation_data.Labels,YPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

save('qweqwe.mat', 'net');
