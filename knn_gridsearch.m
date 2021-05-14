clear all

addpath(genpath('functions/'))
%% Step 0: Putting the paths here for speed

CLASSIFIER = 'nearest neighbor';

data_path = '../data/';

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
    'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
    'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

%number of training examples per category to use. Max is 100.
num_train_per_cat = 100;

fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);


%% Train and test

% knn grid search params
KS = 1:15;              % 1 - 15 values of k (nearest neighbours)
DISTS = {               % distance metrics to calculate nearest neighbours
    'euclidean', 'cityblock', 'minkowski', 'chebychev', 'cosine', 'hamming'
    };


%% bag of sifts gridsearch
% bag of sifts grid search params
CLUSTERS = [50 100 200];    % vocab size
STEPS = [2 3 4 5];          % sift feature step
SPACES = {...               % colour space
    'greyscale', 'hsv', 'rgb'
    };

% grid search
bag_of_sifts_knn_results = [];
for cluster_i=1:length(CLUSTERS)
    cluster = CLUSTERS(cluster_i);
    for step_i=1:length(STEPS)
        step = STEPS(step_i);
        for space_i=1:length(SPACES)
            space = SPACES{space_i};
            for dist_i=1:length(DISTS)
                dist = DISTS{dist_i};
                for k_i=1:length(KS)
                    k = KS(k_i);
                    
                    fprintf('clusters: %d, step: %d, space: %s, dist: %s, k: %d\n', cluster, step, space, dist, k);
                    feats_dir = strjoin(["feats/" int2str(cluster) "clusters/" int2str(step) "step/"], '');
                    feats_path = strjoin([feats_dir space ".mat"], '');
                    
                    load(feats_path);
                    
                    predicted_categories = nearest_neighbour_classify(train_image_feats, train_labels, test_image_feats, k, dist);
                    
                    [accuracy, confusion_matrix] = evaluate(...
                        predicted_categories, ...
                        test_labels, categories ...
                        );
                    
                    bag_of_sifts_knn_results(end+1).feature = 'bag of sifts';
                    bag_of_sifts_knn_results(end).vocab_size = cluster;
                    bag_of_sifts_knn_results(end).sift_step = step;
                    bag_of_sifts_knn_results(end).colour_space = space;
                    bag_of_sifts_knn_results(end).distance = dist;
                    bag_of_sifts_knn_results(end).k = k;
                    bag_of_sifts_knn_results(end).accuracy = accuracy;
                    bag_of_sifts_knn_results(end).cmatrix = confusion_matrix;
                    
                    disp(accuracy);
                end
            end
        end
    end
end

save('bag_of_sifts_knn_results','bag_of_sifts_knn_results')


%% spatial pyramids gridsearch
% spatial pyramids grid search params
CLUSTERS = [50 100 200];    % vocab size
STEPS = [2 3 4 5];          % sift feature step
SPACES = {...               % colour space
    'greyscale', 'hsv', 'rgb'
    };
LAYERS = [1 2 3];

% grid search
spatial_pyramid_knn_results = [];
for cluster_i=1:length(CLUSTERS)
    cluster = CLUSTERS(cluster_i);
    for step_i=1:length(STEPS)
        step = STEPS(step_i);
        for space_i=1:length(SPACES)
            space = SPACES{space_i};
            for layer_i = 1:length(LAYERS)
                layer = LAYERS(layer_i);
                for dist_i=1:length(DISTS)
                    dist = DISTS{dist_i};
                    for k_i=1:length(KS)
                        k = KS(k_i);
                        
                        fprintf('clusters: %d, step: %d, space: %s, layers: %d, dist: %s, k: %d\n', cluster, step, space, layer, dist, k);
                        feats_dir = strjoin(["pyramidfeats/" int2str(cluster) "clusters/" int2str(step) "step/" int2str(layer) "layers/"], '');
                        feats_path = strjoin([feats_dir space ".mat"], '');
                        
                        load(feats_path);
                        
                        predicted_categories = nearest_neighbour_classify(train_image_feats, train_labels, test_image_feats, k, dist);
                        
                        [accuracy, confusion_matrix] = evaluate(...
                            predicted_categories, ...
                            test_labels, categories ...
                            );
                        
                        spatial_pyramid_knn_results(end+1).feature = 'spatial pyramid';
                        spatial_pyramid_knn_results(end).vocab_size = cluster;
                        spatial_pyramid_knn_results(end).sift_step = step;
                        spatial_pyramid_knn_results(end).colour_space = space;
                        spatial_pyramid_knn_results(end).layer = layer;
                        spatial_pyramid_knn_results(end).distance = dist;
                        spatial_pyramid_knn_results(end).k = k;
                        spatial_pyramid_knn_results(end).accuracy = accuracy;
                        spatial_pyramid_knn_results(end).cmatrix = confusion_matrix;
                        
                        disp(accuracy);
                    end
                end
            end
        end
    end
end

save('spatial_pyramid_knn_results', 'spatial_pyramid_knn_results');







