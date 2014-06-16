function [fdatabase] = retr_fdatabase_dir(fea_dir)
%=========================================================================
% inputs
% fea_dir       -the rootpath for the feature database. e.g. '../features/caltech101'
% outputs
% fdatabase      -a tructure of the dir
%                   .path   pathes for each feature file
%                   .label  label for each feature file
% written by gcdofree
% June 2014
%=========================================================================

fprintf('Dir the LLC database...');
subfolders = dir(fea_dir);

fdatabase = [];

fdatabase.label = []; % label of each class
fdatabase.path = {}; % contain the pathes for each image of each class

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        
        frames = dir(fullfile(fea_dir, subname, '*.mat'));
        c_num = length(frames);
                    
        fdatabase.label = [fdatabase.label; ones(c_num, 1)*str2num(subname)];
        
        for jj = 1:c_num,
            c_path = fullfile(fea_dir, subname, frames(jj).name);
            fdatabase.path = [fdatabase.path, c_path];
        end;
    end;
end;
fprintf('Done!\n');