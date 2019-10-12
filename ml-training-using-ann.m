  
hepatitis_data=readtable('hepatitis_2_csv.csv');

hepatitis=table2array(hepatitis_data);

% Training Data
training_class_name = hepatitis(1:520,20:20);
training_data=hepatitis(1:520,1:19);

% Validating Data
validating_class_names = hepatitis(521:568,20:20);
validating_data = hepatitis(521:568,1:19);

% Input for ANN  
tr1=transpose(training_data);
theta1=randperm(10000,19);
inputs1=theta1*tr1;
inputs1=transpose(inputs1);

tr2=transpose(validating_data);
theta2=randperm(10000,19);
inputs2=theta2*tr2;
inputs2=transpose(inputs2);

% Applying ANN
net = feedforwardnet(10);
net = train(net,inputs1,training_class_name);
view(net)
y = net(inputs2);

% Knowing the performance
performance = perform(net,y,validating_class_names)
