function extr_sift(img_dir, data_dir)
% for example
% img_dir = 'image/Caltech101';
% data_dir = 'data/Caltech101';

fprintf('Extract SIFT feature...\n');

addpath('sift');

% you can change the parameters for SIFT as shown below
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;        % image size larger than maxImSize will be resize to maxImSize 
nrml_threshold = 1;

[database, lenStat] = CalculateSiftDescriptor(img_dir, data_dir, gridSpacing, patchSize, maxImSize, nrml_threshold);
fprintf('Done!\n');