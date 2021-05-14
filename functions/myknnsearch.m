function [indexes] = myknnsearch(g,t,k,metric)
% Reimplementation of the matlab knnsearch algorithm for the coursework
% takes each point in t and searches g for the nearest k neighbours
%
% returns an array of length t*k containing the indexes of g which are the 
%         nearest neighbours to each datapoint of t
%
%   g           ground truth (kNN training data)
%   t           test data
%   k           no. of nearest neighbours to compute
%   metric      distance metric to use

    % are the lengths of the feature vectors the same?
    if size(g,2) ~= size(t,2)
        error('The number of features in the train and test data do not match')
    end
    
    D = pdist2(g, t, metric);
    [~,sorted] = sort(D);
    indexes = sorted(1:k,:);
end

