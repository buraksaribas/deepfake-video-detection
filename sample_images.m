real_imgs = {};
for i=1:4
    cap = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), real_train_samples(i)));
    frame = cap.read(1);
    real_imgs{i} = frame;
end

figure;
for i=1:4
    subplot(2,2,i),imshow(real_imgs{i});
end

fake_imgs = {};
for i=1:4
    cap = VideoReader(fullfile(fullfile(datafolder, train_sample_folder), fake_train_samples(i)));
    frame = cap.read(1);
    fake_imgs{i} = frame;
end

figure;
for i=1:4
    subplot(2,2,i),imshow(fake_imgs{i});
end


