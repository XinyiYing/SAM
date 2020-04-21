clc
clear

load_path = './middlebury/hr/';
file_list = dir([load_path]);

%% hr
path_hr = './hr/'; mkdir(path_hr)
path_lrx2 = './lr_x2/'; mkdir(path_lrx2)
path_lrx4 = './lr_x4/'; mkdir(path_lrx4)

for idx = 3:1:length(file_list)
    file_name = file_list(idx).name;
    img_hr_L = imread([load_path, file_name, './hr0.png']);
    img_hr_R = imread([load_path, file_name, './hr1.png']);
    img_hr_L = img_hr_L(1:floor(size(img_hr_L,1)/16)*16, 1:floor(size(img_hr_L,2)/16)*16, :);
    img_hr_R = img_hr_R(1:floor(size(img_hr_R,1)/16)*16, 1:floor(size(img_hr_R,2)/16)*16, :);
    
    mkdir([path_hr, file_name])
    imwrite(img_hr_L, [path_hr, file_name,'/hr0.png']);
    imwrite(img_hr_R, [path_hr, file_name, '/hr1.png']);
    
    mkdir([path_lrx4, file_name])
    img_lr_L = imresize(img_hr_L, 1/4, 'bicubic');
    img_lr_R = imresize(img_hr_R, 1/4, 'bicubic');
    imwrite(img_lr_L, [path_lrx4, file_name, '/lr0.png']);
    imwrite(img_lr_R, [path_lrx4, file_name, '/lr1.png']);
    
    mkdir([path_lrx2, file_name])
    img_lr_L = imresize(img_hr_L, 1/2, 'bicubic');
    img_lr_R = imresize(img_hr_R, 1/2, 'bicubic');
    imwrite(img_lr_L, [path_lrx2, file_name, './lr0.png']);
    imwrite(img_lr_R, [path_lrx2, file_name, './lr1.png']);
    
end