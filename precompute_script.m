clear all
addpath(genpath('functions/'))
addpath(genpath('pyramidfeats/'))

SIZES = [50, 100];
SPACES = ["greyscale", "rgb", "hsv"];
STEPS = [2 3 4 5];
L = [1 2 3];


data_path = '../data/';
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
    'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
    'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
num_train_per_cat = 100;

[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);


for size_i = 1: length(SIZES)
    size = SIZES(size_i);
    for space_i = 1:length(SPACES)
        space = SPACES(space_i);
        %
        vocab_dir = strjoin(["vocabs/" int2str(size) "clusters/"], '');
        vocab_path = strjoin([vocab_dir space ".mat"], '');
        
        disp(vocab_path);
        if ~exist(vocab_path, 'file')
            disp('building vocab')
            vocab = build_vocabulary(train_image_paths, size, space);
            mkdir(convertStringsToChars(vocab_dir));
            save(vocab_path, 'vocab');
        end
        %
        for step_i = 1:length(STEPS)
            step = STEPS(step_i);
            
            for l_i = 1:length(L)
                layers = L(l_i);
                feats_dir = strjoin(["pyramidfeats/" int2str(size) "clusters/" int2str(step) "step/" int2str(layers) "layers/"], '');
                feats_path = strjoin([feats_dir space ".mat"], '');
                
                disp(feats_path);
                if ~exist(feats_path, 'file')
                    train_image_feats = get_spatial_optimised(train_image_paths, size, space, step, layers);
                    test_image_feats  = get_spatial_optimised(test_image_paths, size, space, step, layers);
                    mkdir(convertStringsToChars(feats_dir));
                    save(feats_path, 'train_image_feats', 'test_image_feats')
                end
            end
        end
    end
end




