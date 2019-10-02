hepatitis_data=readtable('hepatitis_2_csv.csv');

hepatitis=table2array(hepatitis_data);

% training data
training_class_name = hepatitis(1:520,20:20);
training_data = hepatitis(1:520,1:19);

% validating data
validating_class_names = hepatitis(521:568,20:20);
validating_data = hepatitis(521:568,1:19);
while 1==1
    choice=menu('Pick One','        Training        ','       Testing        ','      exit      ');
    if choice==1
        % getting the svm model using fitcsvm function
        SVM_model=fitcsvm(training_data,training_class_name);
    else if choice==2
        % predicting the class names and also the score 
        [predicted_class_name,score] = predict(SVM_model,validating_data); 

        % checking the accuracy 
        accuracy = sum(predicted_class_name == validating_class_names)/numel(validating_class_names)
        cm = confusionchart(validating_class_names,predicted_class_name);
        mean_square_error=mean((validating_class_names-predicted_class_name).^2)
        
        else
            close all;
            break
            clear;
        end
    end
end
