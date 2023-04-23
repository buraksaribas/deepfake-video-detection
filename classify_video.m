vid = VideoReader(fullfile(fullfile(datafolder, train_sample_folder),"abarnvbtwb.mp4"));
detector = mtcnn.Detector();
loaded_network = load('deepfake_squeezenet.mat');
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