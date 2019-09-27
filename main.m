hepatitis=readtable('hepatitis_csv.csv');

% training data
training_class_name = hepatitis(1:110,20:20);
training_data = hepatitis(1:110,1:19);

% validating data
validating_class_names = hepatitis(111:142,20:20);
validating_data = hepatitis(111:142,1:19);

SVM_model=fitcsvm(data,class_name)
