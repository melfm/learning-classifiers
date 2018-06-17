function [matrixImageUpdate] = updateNeighbourhood(matrixImage, x, neighbours, winneri,winnerj,alpha)

    % Copy the original first
    matrixImageUpdate(:,:,1) = matrixImage(:,:,1);
    matrixImageUpdate(:,:,2) = matrixImage(:,:,2);
    matrixImageUpdate(:,:,3) = matrixImage(:,:,3);
    
    for i = 1 : size(neighbours,2)
      % Go thru the list and update the pixel for them
      % Get the pixel to be updated
       location = neighbours{i};
       w_old = squeeze(matrixImage(location(1,1),location(1,2),:));
       diff = transpose(x) - w_old;
       w_new =  w_old + (alpha * diff);

       matrixImageUpdate(location(1,1),location(1,2), 1) = w_new(1,1);
       matrixImageUpdate(location(1,1),location(1,2), 2) = w_new(2,1);
       matrixImageUpdate(location(1,1),location(1,2), 3) = w_new(3,1);
    end
    
    % Now update the neuron itself
    w_old = squeeze(matrixImage(winneri,winnerj,:));
    diff = transpose(x) - w_old;
    w_new =  w_old + (alpha * diff);
    matrixImageUpdate(winneri,winnerj, 1) = w_new(1,1);
    matrixImageUpdate(winneri,winnerj, 2) = w_new(2,1);
    matrixImageUpdate(winneri,winnerj, 3) = w_new(3,1);
    
end