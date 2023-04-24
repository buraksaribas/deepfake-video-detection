dataset = imageDatastore('frames', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');

[training_data, validation_data] = splitEachLabel(dataset, 0.7,'randomized');

net = resnet50;

input_layer_size = net.Layers(1).InputSize(1:2);

resized_training_data = augmentedImageDatastore(input_layer_size, training_data);
resized_validation_data = augmentedImageDatastore(input_layer_size, validation_data);

layers = [
    imageInputLayer([224 224 3])
    net(2:end-3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 200, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationData', resized_validation_data, ...
    'ValidationFrequency', 10, ...
    'ExecutionEnvironment', 'auto');

net = trainNetwork(resized_training_data, layers, options);

YPred = classify(net,resized_validation_data);

figure
cm = confusionchart(validation_data.Labels,YPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
