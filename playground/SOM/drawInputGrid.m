% Generate the input pattern as an image stack
function [imageGrid] = drawInputGrid(colorInput)

    imageGrid = zeros(6,1,3);

    color1 = colorInput{1};

    imageGrid(1,1,1) = color1(1);
    imageGrid(1,1,2) = color1(2);
    imageGrid(1,1,3) = color1(3);

    color2 = colorInput{2};

    imageGrid(2,1,1) = color2(1);
    imageGrid(2,1,2) = color2(2);
    imageGrid(2,1,3) = color2(3);

    color3 = colorInput{3};

    imageGrid(3,1,1) = color3(1);
    imageGrid(3,1,2) = color3(2);
    imageGrid(3,1,3) = color3(3);

    color4 = colorInput{4};

    imageGrid(4,1,1) = color4(1);
    imageGrid(4,1,2) = color4(2);
    imageGrid(4,1,3) = color4(3);

    color5 = colorInput{5};

    imageGrid(5,1,1) = color5(1);
    imageGrid(5,1,2) = color5(2);
    imageGrid(5,1,3) = color5(3);

    color6 = colorInput{6};

    imageGrid(6,1,1) = color6(1);
    imageGrid(6,1,2) = color6(2);
    imageGrid(6,1,3) = color6(3);

end