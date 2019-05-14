function classifier = trainClassifier(features, labels, param)
% TRAINCLASSIFIER trains classifiers
%   CLASSIFIER = TRAINCLASSIFIER(FEATURES, LABELS, PARAM) learns a
%   classifier given the training data and parameters. If
%   PARAM.CLASSIFIER='knn', then a nearest neighbour classifier is learned,
%   and if PARAM.CLASSIFIER='svm', a SVM classifier is learned. For SVM we
%   use a linear SVM solver (primal_svm.m file by Olivier Chapelle).
%
% This code is developed by Subhransu Maji from UMass Amherst
%
% Homework 3: Recognition

classifier.type = param.classifier;
switch param.classifier
    case 'knn'
        classifier.k = param.knn.k; % number of nearest neighbours
        classifier.features = features;
        classifier.labels = labels;
        fprintf('Trained k-nn classifier (k=%i).\n', param.knn.k);        

    case 'svm'
        X = features';
        Y = 2*(labels==1)-1; % +1 for cats, -1 for dogs
        [wts, bias] = primal_svm(X, Y, param.svm.lambda);
        classifier.wts = wts';
        classifier.bias = bias;
        fprintf('Trained svm classifier (lambda=%f).\n', param.svm.lambda);        
end
