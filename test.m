loaded_network = load('all_deepfake_squeezenet.mat');
net = loaded_network.net;
datafolder = "C:\Users\Burak\Desktop\matlab\deepfake_video_detection\dataset";
test_folder = "test_videos";

test_samples = dir(fullfile(fullfile(datafolder, test_folder), "*.mp4"));

test_videos_predict = struct();

for i = 1:numel(test_samples)
    fprintf("%d/%d\n", i,numel(test_samples));
    test_video = test_samples(i);

    input_size = net.Layers(1).InputSize(1:2);
    test_frames = prepareVideo(fullfile(test_video.folder,test_video.name), input_size);

    if ~isempty(test_frames)
        [label, probability] = classify(net, cell2table(test_frames));
        test_videos_predict(i).name = test_video.name; 
        test_videos_predict(i).label = label; 
        test_videos_predict(i).prob = max(probability); 
        fprintf("%s %f\n", label, max(probability)*100);
    else
        test_videos_predict(i).name = test_video.name; 
        test_videos_predict(i).label = "unkown"; 
        test_videos_predict(i).prob = 0; 
        fprintf("%s %f\n", label, max(probability)*100);
    end
end

% T = struct2table(test_videos_predict);
% writetable(T,"custom_model.csv")


for i = 1:numel(test_frames)
    subplot(4,3,i)
    [label, probability] = classify(net, test_frames{i});
    imshow(test_frames{i});
    title(string(label) + ", " + num2str(100*max(probability(1, :)),3) +"%");
end


