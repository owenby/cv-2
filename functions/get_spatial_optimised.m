
% Implementated according to the starter code prepared by James Hays, Brown University
% Michal Mackiewicz, UEA

function image_feats = get_spatial_optimised(image_paths, vocab_size, space, step, num_levels)

% normalise space name and get channel names
[space, channels] = get_spaces(space);

% check if a saved vocab already exists on file
vocab_path = ['vocabs/' int2str(vocab_size) 'clusters/' space '.mat'];
if ~exist(vocab_path, 'file')
    error('Construct a vocabulary for this size and space first using build_vocabulary(image_paths, vocab_size, space).');
end

disp(vocab_path);
vocab = getfield(load(vocab_path), 'vocab');

% how many images
[N, ~] = size(image_paths);
% how big the histogram is
M = round(vocab_size * 1/3 * (4^(num_levels)-1));
% total size
image_feats = zeros(N, M * length(channels));

% Go through every image
for i = 1:N
    
    im = convert_colour_space(imread(image_paths{i}), space);
    [m,n,~] = size(im);
   
    % splits 
    l = num_levels-1;
    % size of the grid of finest histograms
    histogrid = zeros(2^l, 2^l, vocab_size * length(channels));
    xjump = m/ 2^l;
    yjump = n/ 2^l;
    xmin = 1;
    % Going through each block in the split image
    for x=1:2^l
        xmax = floor(xmin + xjump-1);
        ymin = 1;
        for y=1:2^l
            ymax = floor(ymin + yjump-1);
            
            feature_block = zeros(vocab_size * length(channels), 1);
            for c = 1:length(channels)
                % compute image sift features and nearest clusters
                [~, sift_feats] = vl_dsift(im(xmin:xmax,ymin:ymax,c), 'step', step);
                D = vl_alldist2(vocab(:,:,c)', double(sift_feats));
                
                [~, sorted] = sort(D);
                nearest_clusters = sorted(1,:);
                
                % compute nearest cluster histogram
                % preallocate for speed
                buffer = zeros(vocab_size, 1);
                for cl = 1:length(nearest_clusters)
                    buffer(nearest_clusters(cl)) = buffer(nearest_clusters(cl)) + 1;
                end
                
                % normalise to proportional histogram
                buffer = buffer ./ length(nearest_clusters);
                
                % compute feature vector offset for channel feature concatenation
                offset = vocab_size * (c - 1) + 1;
                feature_block(offset:c*vocab_size) = buffer;
            end
            % at this point we have a single block's histogram
            
            histogrid(x, y, :) = feature_block;
            
            ymin = ymax + 1;
        end
        xmin = xmax + 1;
    end
    
    
    % for each courser grid, sum, normalise, then concatinate histograms
    feature_vector = zeros(1, M * length(channels));
    newhistogrid = histogrid;
    hist_count = 1;
    
    while l >= 0
        [side, ~, ~] = size(newhistogrid);
        % concat histo grid
        for x=1:side
            for y=1:side
                current_histo = squeeze(histogrid(x, y, :));
                %feature_vector = [feature_vector; 1/(2^(num_levels-l))*current_histo];
                
                if l>0
                	feature_vector(hist_count:hist_count+(vocab_size * length(channels))-1) = 1/(2^(num_levels-l))*current_histo;
                elseif l==0
                    feature_vector(hist_count:hist_count+(vocab_size * length(channels))-1) = 1/(2^(num_levels-1))*current_histo;
                end
                hist_count = hist_count + (vocab_size * length(channels));
            end
        end

        % create histo grid
        if l > 0
            side = side /2;
            newhistogrid = zeros(side, side, vocab_size * length(channels));
            for x=1:2:side
                for y=1:2:side
                    newhistogrid(x, y, :) = (histogrid(x, y, :) + histogrid(x+1, y, :) + histogrid(x, y+1, :) + histogrid(x+1, y+1, :))/4;
                end
            end
            histogrid = newhistogrid;
        end
        l = l - 1;
    end
    image_feats(i,:) = feature_vector;
end


end


function [space, space_names] = get_spaces(space)
if strcmp(space, "rgb")
    % RGB space names
    space_names = ["red" "green" "blue"];
elseif strcmp(space, "hsv")
    % HSV space names
    space_names = ["hue" "saturation" "value"];
elseif strcmp(space, "ycbcr")
    % YCbCr space names
    space_names = ["Y" "Cb" "Vr"];
elseif strcmp(space, "xyz")
    % XYZ space names
    space_names = ["X" "Y" "Z"];
elseif strcmp(space, "yiq")
    % YIQ space names
    space_names = ["Y" "I" "Q"];
else
    space_names = "greyscale";
    space = 'greyscale';
end

space = convertStringsToChars(space);
end

function im = convert_colour_space(im, space)
% Converts from rgb to another space.
% Can be any of:
%   'hsv'   ->  HSV
%   'ycbcr' ->  YCbCr
%   'xyz'   ->  XYZ (CIE 1976)
%   'yiq'   ->  YIQ (NTSC)

% Any other string passed in the space parameter will result in the RGB
% image being returned unchanged
if strcmp(space, 'rgb')
    % convert to RGB colour space
elseif strcmp(space, 'hsv')
    % convert to HSV colour space
    im = rgb2hsv(im);
elseif strcmp(space, 'ycbcr')
    % convert to YCbCr colour space
    im = rgb2ycbcr(im);
elseif strcmp(space, 'xyz')
    % convert to XYZ colour space
    im = rgb2xyz(im);
elseif strcmp(space, 'yiq')
    % convert to YIQ colour space
    im = rgb2ntsc(im);
else
    im = rgb2gray(im);
end

% ensure that intensities are scaled between 0 and 1
im = im2single(im);
end






