clc
close all 
clear all 
while (1==1)
    choice=menu('Paddy Leaf Disease Detection','....... Training........','....... Testing......','........ Close........');
    if (choice==1)
        xx = 1;
        for gk=[1:2]
           k=gk;
           data=[];
            D = 'C:\Users\GOWTHAM KISHORE\Documents\MATLAB\Train';
            S = dir(fullfile(D,'*.jpg'));
            for n1 = 1:numel(S)
                    F = fullfile(D,S(n1).name);
                    I = imread(F);
                    imshow(I)
                    S(n1).data = I;
                    I = imresize(I,[1000,260]);
                    [I3,RGB] = createMask(I);
                    seg_img = RGB;
                    img = rgb2gray(seg_img);
                    glcms = graycomatrix(img);
                    stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
                    Contrast = stats.Contrast;
                    Energy = stats.Energy;
                    Homogeneity =  stats.Homogeneity;
                    Mean = mean2(seg_img);
                    Standard_Deviation = std2(seg_img);
                    Entropy = entropy(seg_img);
                    RMS = mean2(rms(seg_img));
                    Variance = mean2(var(double(seg_img)));
                    a = sum(double(seg_img(:)));
                    Smoothness = 1-(1/(1+a));
                    m = size(seg_img,1);
                    n = size(seg_img,2);
                    in_diff = 0;
                    for i = 1:m
                        for j = 1:n
                            temp = seg_img(i,j)./(1+(i-j).^2);
                            in_diff = in_diff+temp;
                        end
                    end
                    IDM = double(in_diff);

                    ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
                    disp(ff)
                    disp(k)
                    if k==1
                          Train_Feat = ff;
                    end

                    if k>1
                        Train_Feat = [Train_Feat;ff];
                    end
                    if k<10 && k>1
                        xx = [xx;1];
                    elseif k>1
                        xx = [xx;2];
                    end

                    Train_Label = xx;
            end
        end
         SVM_model=fitcsvm(Train_Feat,Train_Label);
         disp('Train Complete');
    end
    if (choice==2)
        [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
        I = imread([pathname,filename]);
        I = imresize(I,[1000,260]);
        figure, imshow(I); title('Query Leaf Image');
        [I3,RGB] = createMask(I);
        seg_img = RGB;
        figure, imshow(I3); title('BW Image');
        figure, imshow(seg_img); title('Segmented Image');       
        img = rgb2gray(seg_img);
        glcms = graycomatrix(img);
        stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
        Contrast = stats.Contrast;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(seg_img);
        Standard_Deviation = std2(seg_img);
        Entropy = entropy(seg_img);
        RMS = mean2(rms(seg_img));
        Variance = mean2(var(double(seg_img)));
        a = sum(double(seg_img(:)));
        Smoothness = 1-(1/(1+a));
        m = size(seg_img,1);
        n = size(seg_img,2);
        in_diff = 0;
        for i = 1:m
            for j = 1:n
                temp = seg_img(i,j)./(1+(i-j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);
        feat_disease = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
        test = feat_disease;
        [predicted_class_name,score] = predict(SVM_model,feat_disease); 
        accuracy = sum(predicted_class_name == feat_disease)/numel(feat_disease)
%         result = multisvm(Train_Feat,Train_Label,test);
%         if result == 1
%             helpdlg(' Disease Detect ');
%             disp(' Disease Detect ');
%         else
%             helpdlg(' Disease not Detect ');
%             disp('Disease not Detect');
%         end
    end
    if (choice==3)
        close all;
        clear all;
        clc;
        return;
    end
end
