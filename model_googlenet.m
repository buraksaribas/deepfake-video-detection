dataset = imageDatastore('frames', 'IncludeSubFolders', true, 'LabelSource', 'foldernames');

[training_data, validation_data] = splitEachLabel(dataset, 0.7,'randomized');

net = googlenet;
%analyzeNetwork(net)

input_layer_size = net.Layers(1).InputSize(1:2);

resized_training_data = augmentedImageDatastore(input_layer_size, training_data);
resized_validation_data = augmentedImageDatastore(input_layer_size, validation_data);

feature_learner = net.Layers(142).Name;
output_classifier = net.Layers(144).Name;

num_of_classes = numel(categories(training_data.Labels));

new_feature_learner = convolution2dLayer([1,1], num_of_classes, ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10, ...
    'Name', 'Deepfake Feature Learner');

new_classification_layer = classificationLayer('Name', 'Deepfake Classifier');

network_architecture = layerGraph(net);

new_network = replaceLayer(network_architecture,feature_learner, new_feature_learner);
new_network = replaceLayer(new_network,output_classifier, new_classification_layer);

num_classes = numel(categories(training_data.Labels));
max_epochs = 5;
mini_batch_size = 32;
initial_learning_rate = 1e-2;
validation_frequency = floor(numel(resized_training_data.Files)/miniBatchSize);

training_options = trainingOptions('sgdm', ...
    'MiniBatchSize', mini_batch_size, ...
    'MaxEpochs', max_epochs, ...
    'InitialLearnRate', initial_learning_rate, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_validation_data, ...
    'ValidationFrequency', validation_frequency, ...
    'ExecutionEnvironment','parallel',...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(resized_training_data, new_network, training_options);

YPred = classify(net,resized_validation_data);

figure
cm = confusionchart(validation_data.Labels,YPred);
cm.Title = 'Confusion Matrix for Test Data';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

save('all_deepfake_googlenet2.mat', 'net');


