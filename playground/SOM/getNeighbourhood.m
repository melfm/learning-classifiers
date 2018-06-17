% This function calculates a square region of neighbourhood given a radius
% It is important to check for the edges. Since we have a simple square
% shaped neighbourhood, we first determine our allowed range without
% worrying whether it's a valid region or not. We then construct a list of
% nodes and check whether it is within bounds before appending it to the
% list.

function [location_list] = getNeighbourhood(matrixImage, currenti, currentj ,radius)

    % Create a square around the neuron
    
    M = size(matrixImage,1);
    N = size(matrixImage,2);
    
    max_range_row = currenti + radius;
    min_range_row = currenti - radius;
    
    min_range_col = currentj - radius;
    max_range_col = currentj + radius;
   
    location_list = {};
    
    counter_index = 1;
    
    for i = min_range_row : max_range_row
        for j = min_range_col : max_range_col
            if (j <= N && i <= M && j >= 1 && i >=1)
                location_list{counter_index} = [i,j];
                counter_index = counter_index + 1;
            end
        end  
    end
end

