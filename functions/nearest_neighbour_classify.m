function [predictions] = nearest_neighbour_classify(train_feats, train_labels, test_feats, k, metric)
% Nearest neighbour classifier using myknnsearch to find nearest neighbours
%   train_feats     training feature vectors
%   train_labels    training labels
%   test_feats      test feature vectors
%   k               number of nearest neighbours
%   metric          distance metric to calculate nearest neighbours

    nn = myknnsearch(train_feats, test_feats, k, metric);
   
    predictions = cell(size(nn,2), 1);
    for in = 1:size(nn,2)
        n = nn(:,in);
        for ik = k:-1:1
            nearest_labels = unique(train_labels(n(1:ik)));
            freq = zeros(length(nearest_labels), 1);
            for ifreq = 1:length(freq)
              freq(ifreq) = length(find(strcmp(nearest_labels{ifreq}, train_labels(n(1:ik)))));
            end
            [maxval, maxindex] = max(freq);
            n_maxvals = length(find(freq == maxval));
           
            if n_maxvals == 1
                predictions(in) = nearest_labels(maxindex);
                break;
            end
        end
    end
end

