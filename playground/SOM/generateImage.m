% From the input pattern, randomly generate colors.
function matrixImage = generateImage(gridSize,colorInput)

    matrixImage = zeros(gridSize,gridSize,3);
    % Randomize the map's nodes weight vectors
    for i = 1 :gridSize
        for j = 1:gridSize
            idx = randi(numel(colorInput));
            matrixImage(i,j,:) = colorInput{idx};
        end
    end
end