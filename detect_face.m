cap = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), real_train_samples(32)));
frame = cap.read(1);

% faceDetector = vision.CascadeObjectDetector;
% bbox            = step(faceDetector, frame);

detector = mtcnn.Detector();
[bboxes, scores, landmarks] = detector.detect(frame);

if ~isempty(bboxes)
    Imf = insertObjectAnnotation(frame,'rectangle',bboxes,'Faces','linewidth',3);
    imshow(Imf)
    title('detected faces')
else
    position = [0 0];
    label='no face detected';
    imgn = insertText(frame,position,label, 'fontsize',25,'BoxOpacity',1);
    imshow(imgn)
end

for i = 1: size (bboxes)
    croppedFace = imcrop (frame, bboxes (i, :)+5); 
	figure (3);
	subplot (1, 2, i);
	imshow (croppedFace); 
end
