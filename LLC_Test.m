% =========================================================================
% An example code for the algorithm proposed in
%
%   Jinjun Wang, Jianchao Yang, Kai Yu, Fengjun Lv, Thomas Huang, and Yihong Gong.
%   "Locality-constrained Linear Coding for Image Classification", CVPR 2010.
%
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010.
%
% Edited by gcdofree
% Use bagging method to improve the accuracy
% June, 2014.
% =========================================================================

clear all; close all; clc;

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 5;                            % number of neighbors for local coding
c = 10;                             % regularization parameter for linear SVM
                                    % in Liblinear package

nRounds = 10;                       % number of bagging times on the dataset.
tr_num  = 30;                       % training examples per category
mem_block = 5000;                   % maxmum number of testing features loaded
                                    % into memory each time, you can change
                                    % it based on your memory size


% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package as SVM tool

img_dir = 'image/Caltech101';       % directory for the image database                             
data_dir = 'data/Caltech101';       % directory for saving SIFT descriptors
fea_dir = 'features/Caltech101';    % directory for saving final image features

% -------------------------------------------------------------------------
% extract SIFT descriptors, we use Prof. Lazebnik's matlab codes in this package
% change the parameters for SIFT extraction inside function 'extr_sift'
extr_sift(img_dir, data_dir);

% -------------------------------------------------------------------------
% retrieve the directory of the database and load the codebook
% retrieve all SIFT features
database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end

% codebook path. If you use other dataset, you may need to generate your
% own codebook.
Bpath = 'dictionary/Caltech101_SIFT_Kmeans_1024.mat';

load(Bpath);
nCodebook = size(B, 2);              % size of the codebook

% -------------------------------------------------------------------------
dFea = sum(nCodebook*pyramid.^2);
nFea = length(database.path);

% extract LLC features
extr_LLC(nFea, database, fea_dir, B, pyramid, knn);

% retrieve LLC features
fdatabase = retr_fdatabase_dir(fea_dir);

% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

fprintf('\n Testing...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);
bagging_result = [];

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    % load the training features 
    tr_fea = zeros(length(tr_idx), dFea);
    tr_label = zeros(length(tr_idx), 1);
    
    for jj = 1:length(tr_idx),
        fpath = fdatabase.path{tr_idx(jj)};
        load(fpath, 'fea', 'label');
        tr_fea(jj, :) = fea';
        tr_label(jj) = label;
    end
    
    options = ['-c ' num2str(c)];
    model = train(double(tr_label), sparse(tr_fea), options);
    clear tr_fea;

    % load the testing features
    ts_num = length(ts_idx);
    ts_label = [];
    
    if ts_num < mem_block,
        % load the testing features directly into memory for testing
        ts_fea = zeros(length(ts_idx), dFea);
        ts_label = zeros(length(ts_idx), 1);

        for jj = 1:length(ts_idx),
            fpath = fdatabase.path{ts_idx(jj)};
            load(fpath, 'fea', 'label');
            ts_fea(jj, :) = fea';
            ts_label(jj) = label;
        end

        [C] = predict(ts_label, sparse(ts_fea), model);
    else
        % load the testing features block by block
        num_block = floor(ts_num/mem_block);
        rem_fea = rem(ts_num, mem_block);
        
        curr_ts_fea = zeros(mem_block, dFea);
        curr_ts_label = zeros(mem_block, 1);
        
        C = [];
        
        for jj = 1:num_block,
            block_idx = (jj-1)*mem_block + (1:mem_block);
            curr_idx = ts_idx(block_idx); 
            
            % load the current block of features
            for kk = 1:mem_block,
                fpath = fdatabase.path{curr_idx(kk)};
                load(fpath, 'fea', 'label');
                curr_ts_fea(kk, :) = fea';
                curr_ts_label(kk) = label;
            end    
            
            % test the current block features
            ts_label = [ts_label; curr_ts_label];
            [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
            C = [C; curr_C];
        end
        
        curr_ts_fea = zeros(rem_fea, dFea);
        curr_ts_label = zeros(rem_fea, 1);
        curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
        for kk = 1:rem_fea,
           fpath = fdatabase.path{curr_idx(kk)};
           load(fpath, 'fea', 'label');
           curr_ts_fea(kk, :) = fea';
           curr_ts_label(kk) = label;
        end  
        
        ts_label = [ts_label; curr_ts_label];
        [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model); 
        C = [C; curr_C];        
    end
    
    % add to bagging result
    bagging_result = [bagging_result, C];
    
    % normalize the classification accuracy by averaging over different
    % classes
    acc = zeros(nclass, 1);

    for jj = 1 : nclass,
        c = clabel(jj);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end

    accuracy(ii) = mean(acc); 
    fprintf('Classification accuracy for round %d: %f\n', ii, accuracy(ii));
end

Ravg = mean(accuracy);                  % average recognition rate
Rstd = std(accuracy);                   % standard deviation of the recognition rate

% -------------------------------------------------------------------------
% generate the bagging result
[height, width]=size(bagging_result);
final_result = zeros(height, 1);
for ii = 1:height,
    category_result = zeros(nclass, 1);
    for jj = 1:width,
        category_result(bagging_result(ii, jj)) = category_result(bagging_result(ii, jj)) + 1;
    end
    % find majority
    max_num = max(category_result);
    for kk = 1:nclass,
        if(max_num == category_result(kk)),
            final_result(ii) = kk;
            break;
        end
    end
end

acc = zeros(nclass, 1);

% evaluate the bagging result
for jj = 1 : nclass,
    c = clabel(jj);
    idx = find(ts_label == c);
    curr_pred_label = final_result(idx);
    curr_gnd_label = ts_label(idx);    
    acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
end

Rbagging = mean(acc); 

fprintf('===============================================\n');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('Bagging classification accuracy: %f\n', Rbagging);
fprintf('===============================================');
    
