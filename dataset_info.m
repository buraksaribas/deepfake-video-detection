clc;clear;close all;

datafolder = "C:\Users\Burak\Desktop\matlab\deepfake_video_detection\dataset";
train_sample_folder = "train_sample_videos";
test_folder = "test_videos";

train_samples = numel(dir(fullfile(fullfile(datafolder, train_sample_folder), "*.mp4")));
test_samples = numel(dir(fullfile(fullfile(datafolder, test_folder), "*.mp4")));

metadata = fullfile(datafolder, train_sample_folder, "metadata.json");

fid = fopen(metadata); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

fields = fieldnames(val);

fprintf("\t\t\t    label  |  split  |  original\n");
fprintf("----------------------------------------------------\n");
for i = 1:size(fields)
    fprintf("%s  %s \t  %s \t%s\n", fields{i}, val.(fields{i}).label, val.(fields{i}).split, val.(fields{i}).original);
end

fake = 0;
real = 0;

fake_train_samples = {};
real_train_samples = {};

for i = 1:size(fields)
    if(strcmp(val.(fields{i}).label, "FAKE"))
        fake = fake + 1;
        fake_train_samples{end+1} = strrep(fields{i},'_','.');
    else
        real = real + 1;
        real_train_samples{end+1} = strrep(fields{i},'_','.');
    end 
end

y = [fake, real];
x = categorical(["Fake","Real"]);
x = reordercats(x,["Fake","Real"]);
bar(x,y);
title("The label in the training set");
xlabel("Videos");



