function [accuracy,confusion_matrix] = evaluate(predicted_categories, test_labels, categories)
%GET_RESULTS Summary of this function goes here
%   Detailed explanation goes here

    num_categories = length(categories);
    confusion_matrix = zeros(num_categories, num_categories);
    for cat_i=1:length(predicted_categories)
        row = find(strcmp(test_labels{cat_i}, categories));
        column = find(strcmp(predicted_categories{cat_i}, categories));
        confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
    end
    %if the number of training examples and test casees are not equal, this
    %statement will be invalid.
    num_test_per_cat = length(test_labels) / num_categories;
    confusion_matrix = confusion_matrix ./ num_test_per_cat;   
    accuracy = mean(diag(confusion_matrix));
end

