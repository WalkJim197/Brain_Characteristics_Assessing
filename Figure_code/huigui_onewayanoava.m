clc
clear all

%load('F:\project\AD network properties\data\T1Img\TIV.txt');
%save('TIV.mat','TIV');

%load('F:\project\AD network properties\data\T1Img\TIV.mat');
%[a,b] = size(TIV);
%GM_Volume =TIV(:,2);
%WM_Volume =TIV(:,3);
%save('GM_Volume.mat','GM_Volume');
%save('WM_Volume.mat','WM_Volume');

% Group = xlsread('C:\Users\PC\Desktop\mm.xlsx');
% save Group;

load('F:\project\SFC\all_code\Group.mat');

% regress_var=xlsread('F:\project\SFC\人口信息统计表\201_SFC_regress.xlsx');
% save 201_regress_var;

load('F:\project\SFC\all_code\201_regress_var.mat');

% load('F:\project\SFC\T1Img\TIV.txt')
% GM_Volume = TIV(:,2);
% save GM_Volume;
% WM_Volume = TIV(:,3);
% save WM_Volume;

load('F:\project\SFC\T1Img\GM_Volume.mat');

Y_matrix = GM_Volume;%one way anova 分析的对象
[m,n] = size(Y_matrix);

for i = 1:n
     %regression
    Y= Y_matrix(:,i);
    
    [b,bint,r,rint,stats]= regress(Y,regress_var);
    final_r = r +  mean(Y(:));
    X = final_r;
    
%     HC = final_r(1:100);
%     VMCI = final_r(101:190);
%     MCI = final_r(191:243);
    
%     n_HC=repmat({'0'},100,1);
%     n_VMCI=repmat({'1'},90,1);
%     n_MCI=repmat({'2'},53,1);

%     X=[new_HC_data; new_BDD_data; new_MDD_data];
%     Group=[ones(length(new_HC_data),1); 2*ones(length(new_BDD_data),1); 3*ones(length(new_MDD_data),1)];
%     p=anova1(final_r,group);
 
%非参数one way anova检验  
    [p,table,stats] = kruskalwallis(X,Group,'off');
    chi=cell2mat(table(2,5));
    
    A(i) = p;
end


index= find(A<0.05);%找到p<0.05所对应的ROI编号；
p_value=A(index)
data = [index', p_value']
[m, n] = size(data);

data_cell = mat2cell(data, ones(m,1), ones(n,1));
title = {'index', 'p_value'};    
result = [title; data_cell];   
s = xlswrite('GM_Volume.xls', result); % 将result写入到wind.xls文件中

    



