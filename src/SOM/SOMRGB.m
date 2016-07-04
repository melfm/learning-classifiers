%% Self-organizing maps 
%learn to cluster data based on similarity

% Pick a color from input to learn
% Normalized RGB Values
color1 = [0.9, 0.9 ,0];
color2 = [0.8, 0, 0.7];
color3 = [0.9 ,0 , 0];
color4 = [0, 0.1, 0.6];
color5 = [0, 0.5 ,0];
color6 = [0, 0.9, 0.9];

colorInput = {color1;color2;color3;color4;color5;color6};

gridSize = 44;
% Randomize the map's nodes weights
matrixImage = generateImage(gridSize,colorInput);
imageInput = drawInputGrid(colorInput);

% Total number of iteration
T = 100;

figure(1);
imshow(matrixImage,'InitialMagnification','fit')
title('Randomized Weight Map')
figure(2);
imshow(imageInput,'InitialMagnification','fit')
title('Input Pattern')

% Initialize
winneri = 1;
winnerj = 1;
% Initial learning rate
alpha_0 = 0.8;
k = gridSize / 2.0;
alpha = alpha_0;
alphas = {};
ks = {};

for epoch = 1 : T
    if (mod(epoch, 2) == 0)
      % Shrink k
      k = k - 1;
    end
    % Store k for plotting later
    ks{epoch} = k;
    for i = 1 : size(colorInput)
        % Choose an input pattern
        learn = colorInput{i};
        r = learn(1);
        g = learn(2);
        b = learn(3);
        % Select the winning unit
        [winneri, winnerj] = getWinner(matrixImage, r, g, b);
        
        location_list = getNeighbourhood(matrixImage, winneri, winnerj , k);
        % Update the weights
        updatedGrid = updateNeighbourhood(matrixImage,learn,location_list,winneri,winnerj,alpha);

        % Update the map
        matrixImage(:,:,1) = updatedGrid(:,:,1);
        matrixImage(:,:,2) = updatedGrid(:,:,2);
        matrixImage(:,:,3) = updatedGrid(:,:,3);
    end 
    % Update learning rate
    if (epoch < 90)
        alpha = alpha_0 * (1-(epoch/(T+1)));   
    end
    alphas{epoch} = alpha;
    if (epoch == 20)
        % Store the first round of training
        twentyRoundMap(:,:,1) = matrixImage(:,:,1);
        twentyRoundMap(:,:,2) = matrixImage(:,:,2);
        twentyRoundMap(:,:,3) = matrixImage(:,:,3);
    end
    if (epoch == 40)
        % Store the second round of training
        fortyRoundMap(:,:,1) = matrixImage(:,:,1);
        fortyRoundMap(:,:,2) = matrixImage(:,:,2);
        fortyRoundMap(:,:,3) = matrixImage(:,:,3);
    end
    if (epoch == 100)
        % Store the last round of training
        hundredRoundMap(:,:,1) = matrixImage(:,:,1);
        hundredRoundMap(:,:,2) = matrixImage(:,:,2);
        hundredRoundMap(:,:,3) = matrixImage(:,:,3);
    end
end

figure(3)
imshow(twentyRoundMap,'InitialMagnification','fit')
title('Map after 20 epochs')
figure(4)
imshow(fortyRoundMap,'InitialMagnification','fit')
title('Map after 40 epochs')
figure(5)
imshow(hundredRoundMap,'InitialMagnification','fit')
title('Map after 100 epochs')


x = [1: 101];
figure(6);
title('Alpha learning rate decay')
hold on;
for i = 1 : size(alphas,2)
   plot(x(i), cell2mat(alphas(i)), 'ro'); 
   hold on;
end
hold off

alphas = {};

y = [1: 101];
figure(7);
title('Size of neighbourhood (Nc) shrinking by 1 for every two epochs.')
hold on;
for i = 1 : size(ks,2)
   plot(y(i), cell2mat(ks(i)), 'ro'); 
   hold on;
end
hold off

ks = {};

