loaded_network = load('deepfake_classifier.mat');
net = loaded_network.trainedNetwork;

[label, Probability] = classify(net, resized_validation_data);
accuracy = mean(label == validation_data.Labels);

index = randperm(numel(validation_data.Files),4);

figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(validation_data, index(i));
    imshow(I)
    title(string(label(index(i))) + ", " + num2str(100*max(Probability(index(i), :)),3) +"%");
end
    