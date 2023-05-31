%laod iris
load fisheriris;

trainData = meas(:,3:4);
trainLabel = species;
Y = categorical (species);

dtClass = categories(Y);

x = trainData (:,1);
y = trainData (:,2);
gscatter(x,y,trainLabel,'rgb','osd');


xlabel ('Petal length');
ylabel('Petal width');


%model
tree = ClassificationTree.fit (trainData,trainLabel);
view (tree, 'mode', 'graph');


testData = trainData;
testLabel = species;

[dtClass, score] = predict(tree,testData);
dtaccuracy = sum(strcmp(dtClass, trainLabel))/numel(trainLabel);

disp(['DT Accuracy:' num2str(dtaccuracy)]);

tabulate(trainLabel);
NB = fitcnb(trainData,trainLabel);
[nbClass, Posterior,Cost] = predict(NB,testData);


NBaccuracy = sum(strcmp(nbClass, trainLabel)) /numel (trainLabel);
disp(['NB Accuracy:' num2str(NBaccuracy)]);

message = sprintf ('Classifier''s accuracy: \n DT = %.2f%% \n NB = %.2f%%'...
    ,dtaccuracy, NBaccuracy);
msgbox(message, 'modal');

