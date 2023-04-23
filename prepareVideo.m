function [testVideoFrames] = prepareVideo(file, input_size)
    detector = mtcnn.Detector();
    capture_image = VideoReader(file);
    numFrames = capture_image.NumFrames;
    frames = {};
    count = 1;
    
    for i = 1:60:numFrames
        frame = read(capture_image,i);
            
        [bboxes, scores, landmarks] = detector.detect(frame);
        if ~isempty(bboxes)
            for j = 1: size (bboxes)
                croppedFace = imcrop (frame, bboxes (j, :)+5);
                frames{count} = imresize(croppedFace, input_size);
                count = count + 1;
            end
        end
    end
    testVideoFrames = frames
end

