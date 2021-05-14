function predictions = cnn_classify(train_image_paths, test_image_paths, transfer_model)
% creates and trains a CNN classifier on the images passed in
% train_image_paths
    
    if strcmp(transfer_model, 'googleplaces')
        net = googlenet('Weights', 'places365');
        input_size = [224 224 3];
    elseif strcmp(transfer_model, 'resnet18')
        net = resnet18;
        input_size = [224 224 3];
    elseif strcmp(transfer_model, 'resnet50')
        net = resnet50;
        input_size = [224 224 3];
    end
    
        
    og_train_imds = imageDatastore(train_image_paths, 'IncludeSubfolders',true,'LabelSource','foldernames');
    og_test_imds = imageDatastore(test_image_paths, 'IncludeSubfolders',true,'LabelSource','foldernames');
    
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-5 5], ...
        'RandXScale', [1 1.25], ...
        'RandYScale', [1 1.22], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5] ...
    );
    
        
    train_imds = augmentedImageDatastore(input_size, og_train_imds, 'DataAugmentation', augmenter);
    test_imds = augmentedImageDatastore(input_size, og_test_imds);
        
    lgraph = layerGraph(net);
    [learnableLayer, classLayer] = findLayersToReplace(lgraph);
    
    % replace classification layers for a 15 class problem
    new_fc = fullyConnectedLayer(15, 'Name', 'new_fc');
    lgraph = replaceLayer(lgraph, learnableLayer.Name, new_fc);
    new_output = classificationLayer('Name', 'new_output');
    lgraph = replaceLayer(lgraph, classLayer.Name, new_output);
    
    % freeze first layers to prevent overfitting to transfer set
    layers = lgraph.Layers;
    connections = lgraph.Connections;
    
    layers(1:10) = freezeWeights(layers(1:10));
    lgraph = createLgraphUsingConnections(layers, connections);
    
    options = trainingOptions('sgdm', ...
                              'MaxEpochs',10,...
                              'InitialLearnRate',3e-4, ...
                              'Verbose',false, ...
                              'Plots','training-progress', ...
                              'ExecutionEnvironment', 'gpu', ...
                              'ValidationData', test_imds);
    
    net = trainNetwork(train_imds, lgraph, options);
    predictions = cellstr(classify(net, test_imds));
end



function layers = freezeWeights(layers)
% layers = freezeWeights(layers) sets the learning rates of all the
% parameters of the layers in the layer array |layers| to zero.
    for ii = 1:size(layers,1)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end
end

function lgraph = createLgraphUsingConnections(layers,connections)
% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph,layers(i));
    end

    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
    end
end

