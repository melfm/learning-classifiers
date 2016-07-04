% This function finds the BMU by using Euclidean distance formula
% to find the similarity between the input vector (decomposed as
% r, g , b components and the maps node weight vector.

function [winneri, winnerj] = getWinner(matrixImage, r, g ,b)
    winneri = 0;
    winnerj = 0;
    % Initialize to some large value
    closest = sqrt((255.0 * 255.0) + (255.0 * 255.0) + (255.0 * 255.0));
    for i = 1 : size(matrixImage,2)
        for j = 1 : size(matrixImage,2)
           % Calculate the euclidean distance of the current neuron
 
           diff_r = (matrixImage(i,j,1) - r)^2;
           diff_g = (matrixImage(i,j,2) - g)^2;
           diff_b = (matrixImage(i,j,3) - b)^2;

           distance = sqrt((diff_r + diff_g + diff_b));
           
           if (distance < closest)
               winneri = i;
               winnerj = j;
               closest = distance;
           end
        end
    end   
end