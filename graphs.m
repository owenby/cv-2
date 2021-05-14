
load('spatial_pyramid_svm_results.mat')

load('bag_of_sifts_svm_results.mat')

%a = bag_of_sifts_knn_results;
%a = spatial_pyramid_knn_results;

%a = bag_of_sifts_svm_results;
a = spatial_pyramid_svm_results;



x = [];

for i=1:length(a)
    
    acc = a(i).accuracy;
    
    %x = [x; acc];
    if a(i).layer == 3
        x = [x; acc];
    end

end

mean(x)



% mean layer 1 = 0.3719