real_imgs = {};
fake_imgs = {};
for i=1:4
    capR = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), real_train_samples(i)));
    frameR = capR.read(1);
    
    capF = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), fake_train_samples(i)));
    frameF = capF.read(1);
    
    real_imgs{i} = frameR;
	fake_imgs{i} = frameF;
end

figure;
for i=1:4
    subplot(2,2,i),imshow(real_imgs{i});
end

figure;
for i=1:4
    subplot(2,2,i),imshow(fake_imgs{i});
end



