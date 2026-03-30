clc; clear; close all;

%% ==================== SECTION 1: Image Visualization & Processing ====================

% Local images for demonstration
imagePaths = {
    'resized_Y11.jpg', 
    'resized_18_no.jpg', 
    'resized_Y50.JPG', 
    'resized_Y65.JPG'
};

numImages = numel(imagePaths);

% Initialize
images = cell(1,numImages);
grayImages = cell(1,numImages);
resizeImages = cell(1,numImages);
J = cell(1,numImages);
segmented = cell(1,numImages);
edges = cell(1,numImages);

% Process images
for i = 1:numImages
    images{i} = imread(imagePaths{i});
    
    % Convert to grayscale
    if size(images{i},3) == 3
        grayImages{i} = rgb2gray(images{i});
    else
        grayImages{i} = images{i};
    end

    % Resize
    resizeImages{i} = imresize(grayImages{i}, [256, 256]);

    % Guided filtering
    range = [min(resizeImages{i}(:)), max(resizeImages{i}(:))];
    smoothValue = 0.01 * diff(range).^2;
    J{i} = imguidedfilter(resizeImages{i}, 'DegreeOfSmoothing', smoothValue);

    % Segmentation using Otsu
    thresh = multithresh(J{i}, 2);
    segmented{i} = imquantize(J{i}, thresh);
    segmented{i} = imfill(segmented{i}, 'holes');

    % Edge detection
    edges{i} = edge(resizeImages{i}, 'Canny');
end

% Display results
for i = 1:numImages
    figure;

    subplot(2,3,1); imagesc(J{i}); colormap(hot); colorbar; title(['Guided Filtered ' num2str(i)]);
    subplot(2,3,2); imshow(label2rgb(segmented{i})); title(['Segmented ' num2str(i)]);
    subplot(2,3,3); imshow(edges{i}); title(['Edges ' num2str(i)]);
    subplot(2,3,4); imagesc(resizeImages{i}); colormap(hot); colorbar;
    tempF = mean(resizeImages{i}(:)) * 0.1 + 32;
    text(50,50,sprintf('%.2f F', tempF), 'Color','cyan', 'FontSize',12);
    title(['Temperature Map ' num2str(i)]);

    subplot(2,3,5); imshow(label2rgb(segmented{i})); hold on;
    stats = regionprops(segmented{i}, 'BoundingBox', 'PixelIdxList');
    for j = 1:length(stats)
        regionPixels = resizeImages{i}(stats(j).PixelIdxList);
        regionTempF = mean(regionPixels) * 0.1 + 32;
        bbox = stats(j).BoundingBox;
        centerX = bbox(1) + bbox(3)/2;
        centerY = bbox(2) + bbox(4)/2;
        text(centerX, centerY, sprintf('%.2f F', regionTempF), 'Color','white', 'FontSize',10,'HorizontalAlignment','center');
    end
    title(['Regions with Temp ' num2str(i)]);
end

%% ==================== SECTION 2: CNN-Based Brain Stone Classification ====================

% Dataset paths
tumorFolder = 'C:\Users\asus\OneDrive\Desktop\brain_tumor_dataset\FInal file\yes';
nonTumorFolder = 'C:\Users\asus\OneDrive\Desktop\brain_tumor_dataset\FInal file\no';

% Image datastores
tumorImages = imageDatastore(tumorFolder, 'LabelSource', 'foldernames');
nonTumorImages = imageDatastore(nonTumorFolder, 'LabelSource', 'foldernames');

% Combined datastore
allImages = imageDatastore([tumorImages.Files; nonTumorImages.Files], ...
    'Labels', [tumorImages.Labels; nonTumorImages.Labels]);
allImages.ReadFcn = @(filename) imresize(imread(filename), [224,224]);

% Split data
[trainData, testData] = splitEachLabel(allImages, 0.8, 'randomized');

% Oversample minority class
labelCounts = countcats(trainData.Labels);
[minorityCount, idx] = min(labelCounts);
majorityCount = max(labelCounts);

if minorityCount < majorityCount
    minorityLabel = categories(trainData.Labels);
    minorityLabel = minorityLabel{idx};
    idxMinority = find(trainData.Labels == minorityLabel);
    numToAdd = majorityCount - minorityCount;
    oversampleIdx = idxMinority(randi(minorityCount, [1, numToAdd]));
    oversampleFiles = trainData.Files(oversampleIdx);
    oversampleLabels = repmat(categorical({minorityLabel}), numToAdd,1);
    trainData = imageDatastore([trainData.Files; oversampleFiles], ...
        'Labels', [trainData.Labels; oversampleLabels]);
end

% Data augmentation
augmenter = imageDataAugmenter( ...
    'RandRotation', [-30 30], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandScale', [0.9 1.1]);

augmentedTrain = augmentedImageDatastore([224 224], trainData, ...
    'ColorPreprocessing','gray2rgb','DataAugmentation',augmenter);
augmentedTest = augmentedImageDatastore([224 224], testData, ...
    'ColorPreprocessing','gray2rgb');

%% Standardized CNN Architecture
layers = [
    imageInputLayer([224 224 3],'Name','input')

    convolution2dLayer(3,32,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3,64,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')

    convolution2dLayer(3,128,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    maxPooling2dLayer(2,'Stride',2,'Name','pool3')

    convolution2dLayer(3,256,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4')
    reluLayer('Name','relu4')
    maxPooling2dLayer(2,'Stride',2,'Name','pool4')

    dropoutLayer(0.5,'Name','dropout')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu5')
    fullyConnectedLayer(2,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

%% Train Options
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize',32, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augmentedTest, ...
    'ValidationFrequency',30, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress');

%% Train Network
[net, trainInfo] = trainNetwork(augmentedTrain, layers, options);

%% Evaluate Model
predicted = classify(net, augmentedTest);
actual = testData.Labels;

accuracy = sum(predicted == actual) / numel(actual);
fprintf('\nTest Accuracy: %.2f%%\n', accuracy*100);

trainAcc = trainInfo.TrainingAccuracy(end);
valAcc = trainInfo.ValidationAccuracy(end);
fprintf('Final Train Accuracy: %.2f%%\n', trainAcc);
fprintf('Final Val Accuracy: %.2f%%\n', valAcc);

% Confusion Matrix + Metrics
cm = confusionmat(actual, predicted);
disp('Confusion Matrix:'); disp(cm);

tp = cm(2,2); tn = cm(1,1); fp = cm(1,2); fn = cm(2,1);
precision = tp / (tp + fp + eps);
recall = tp / (tp + fn + eps);
f1 = 2 * (precision * recall) / (precision + recall + eps);

fprintf('Precision: %.2f%%\n', precision*100);
fprintf('Recall: %.2f%%\n', recall*100);
fprintf('F1 Score: %.2f%%\n', f1*100);

%% Accuracy Plot
figure;
plot(trainInfo.TrainingAccuracy,'-o'); hold on;
plot(trainInfo.ValidationAccuracy,'-s');
yline(accuracy*100,'--r','Test Accuracy');
xlabel('Epoch'); ylabel('Accuracy (%)');
legend('Train','Validation','Test');
title('Accuracy vs Epochs'); grid on;