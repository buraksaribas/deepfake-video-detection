test_samples = dir(fullfile(fullfile(datafolder, test_folder), "*.mp4"));

vid = VideoReader(fullfile(fullfile(datafolder, test_folder), test_samples(1).name));
detector = mtcnn.Detector();
loaded_network = load('alexnet.mat');
net = loaded_network.net;

while hasFrame(vid)
    frame = readFrame(vid);
    % do some processing on the frame
    [bboxes, scores, landmarks] = detector.detect(frame);
    
    if ~isempty(bboxes)
        
        face = imcrop(frame,bboxes);
        face = imresize(face, [227 227]);
        class = classify(net, face);
        IFaces = insertObjectAnnotation(frame,'rectangle',bboxes,class,'linewidth',3, FontSize=18);
        
        imshow(IFaces)
    end
end