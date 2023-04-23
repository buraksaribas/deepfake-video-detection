dataset = imageDatastore('frames10', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');

[training_data, validation_data, test_data] = splitEachLabel(dataset, 0.7, 0.15, 0.15, 'randomized');

net = squeezenet;

input_size = net.Layers(1).InputSize(1:2);

resized_training_data = augmentedImageDatastore(input_size, training_data);
resized_validation_data = augmentedImageDatastore(input_size, validation_data);
resized_test_data = augmentedImageDatastore(input_size, test_data);

network_architecture = layerGraph(net);

num_classes = numel(categories(training_data.Labels));
max_epochs = 10;
mini_batch_size = 32;
initial_learning_rate = 1e-1;
validation_frequency = floor(numel(resized_training_data.Files)/mini_batch_size);
    
new_feature_learner_layer = convolution2dLayer([1,1], num_classes, ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10, ...
    'Name', 'Facial Feature Learner');

new_classification_layer = classificationLayer('Name', 'DeepFake Face Classifier');

new_network = replaceLayer(network_architecture,'conv10', new_feature_learner_layer);
new_network = replaceLayer(new_network,'ClassificationLayer_predictions', new_classification_layer);

training_options = trainingOptions('adam', ...
    'MiniBatchSize', mini_batch_size, ...
    'MaxEpochs', max_epochs, ...
    'InitialLearnRate', initial_learning_rate, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_training_data, ...
    'ValidationFrequency', validation_frequency, ...
    'Verbose', false, ...
    'ExecutionEnvironment','parallel',...
    'Plots', 'training-progress');

net = trainNetwork(resized_training_data, new_network, training_options);

pred = classify(net,resized_test_data);

figure
cm = confusionchart(test_data.Labels,pred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

save('squeezenet10.mat', 'net');


