% Based on James Hays, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, lambda)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in coursework_starter,
%because unique() sorts them. This shouldn't really matter, though.
    labels = categorical(train_labels); 
    
    onehot_labels = categories(labels);
    onehot = onehotencode(labels,2);
    onehot(onehot == 0) = -1;
    
    [len, ~] = size(test_image_feats);
    confidences = zeros(len, length(onehot_labels));
    
%     disp(size(confidences));
%     disp(size(train_image_feats));
%     disp(size(onehot(:,1)));
    
    for c=1:length(onehot_labels)
        [w, b] = vl_svmtrain(train_image_feats', onehot(:,c), lambda);
%         [~,~,~, scores] = vl_svmtrain(test_image_feats, y, 0, 'model', w, 'bias', b, 'solver', 'none') ;
        scores = w' * test_image_feats' + b;
%         disp(size(scores));
        
        confidences(:,c) = scores;
%         for j=1:length(train_labels)
%             
%         end
    end
    
%     disp(confidences);
    [~, predicted_index] = max(confidences,[],2);
%     disp(predicted_index);
    predicted_categories = onehot_labels(predicted_index);
%     disp(predicted_categories);
end