datafolder = "C:\Users\Burak\Desktop\matlab\deepfake_video_detection\dataset";
train_sample_folder = "train_sample_videos";

metadata = fullfile(datafolder, train_sample_folder, "metadata.json");

fid = fopen(metadata); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

fields = fieldnames(val);

datafolder = "C:\Users\Burak\Desktop\matlab\deepfake_video_detection\dataset";
train_sample_folder = "train_sample_videos";
test_folder = "test_videos";

detector = mtcnn.Detector();

frameFolder = 'C:\Users\Burak\Desktop\matlab\deepfake_video_detection\frames10';

for i = 1:length(fields)
    capture_image = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), strrep(fields{i},'_','.')));
    numFrames = capture_image.NumFrames;

    fprintf('%d/%d\n', i,length(fields));
    cnt = 1;
    
    for j = 1:30:numFrames
        frame = read(capture_image,j);
        
        [bboxes, scores, landmarks] = detector.detect(frame);
        if ~isempty(bboxes)
            for k = 1: size (bboxes)                
                file_name = sprintf('%s%03d.jpg',extractBefore(fields{i}, "_"), cnt);
                if val.(fields{i}).label == "FAKE"
                    imgName = fullfile(frameFolder,"fake",file_name);    
                else
                    imgName = fullfile(frameFolder,"real",file_name);    
                end
                
                cnt = cnt + 1;
                croppedFace = imcrop (frame, bboxes (k, :)+5);
                imwrite(croppedFace, imgName)
            end
        else
            fprintf("No face %s\n", fields{i});
        end
    end
end


