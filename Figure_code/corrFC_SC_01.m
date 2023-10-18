clc
clear all

% FC_HIP_R_TP = xlsread('F:\project\SFC\SFC_paper_submit\response2comments\FC_HIP_R_TP_99.xlsx');
% save FC_HIP_R_TP;
% mean_MD_R_hipp_TP = xlsread('F:\project\SFC\SFC_paper_submit\response2comments\mean_MD_R_hipp_TP_99.xlsx');
% save mean_MD_R_hipp_TP;
% regress_var_99 = xlsread('F:\project\SFC\人口信息统计表\99_age_sex_edu.xlsx');
% save regress_var_99;

%%准备数据
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\数据被回归了的\mean_FC_HIP_R_TP_cor.mat');
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\数据被回归了的\mean_MD_HIP_R_TP_cor.mat');
mean_FC_HIP_R_TP_cor = r_mean_FC_HIP_R_TP;
mean_MD_HIP_R_TP_cor = r_mean_MD_HIP_R_TP;

HC_FC_data = mean_FC_HIP_R_TP_cor(1:43);
VMCI_FC_data = mean_FC_HIP_R_TP_cor(44:77);
MCI_FC_data = mean_FC_HIP_R_TP_cor(78:99);

HC_SC_data = mean_MD_HIP_R_TP_cor(1:43);
VMCI_SC_data = mean_MD_HIP_R_TP_cor(44:77);
MCI_SC_data = mean_MD_HIP_R_TP_cor(78:99);

load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\数据未回归\mean_FC_HIP_R_TP.mat');
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\数据未回归\mean_MD_HIP_R_TP.mat');


HC_FC_data = mean_FC_HIP_R_TP(1:43);
VMCI_FC_data = mean_FC_HIP_R_TP(44:77);
MCI_FC_data = mean_FC_HIP_R_TP(78:99);

HC_SC_data = mean_MD_HIP_R_TP(1:43);
VMCI_SC_data = mean_MD_HIP_R_TP(44:77);
MCI_SC_data = mean_MD_HIP_R_TP(78:99);

%%计算相关，得到斜率和P值
[r,p] = corr(HC_FC_data,HC_SC_data);
% figure;

% %%画散点图
% scatter(HC_FC_data,HC_SC_data);  
% hold on;
% 
% Y = HC_SC_data;  
% X = [ones(length(HC_FC_data), 1), HC_FC_data];
% b= regress(Y,X);
% x = HC_FC_data;
% y = X*b;
% plot(x,y)
% hold on;

%%gramm 绘图
%导入用于绘制图形的数据
X = HC_FC_data;
Y = HC_SC_data;  
stats = ScatterOutliers(X,Y);  %抛点

h=figure;
set(h,'units','normalized','position',[0.1 0.1 0.4 0.6]); %设置绘图窗口的大小[0.1 0.1 0.4 0.6]
set(h,'color','w'); %设置绘图窗口的背景为白色
color_point=[0.18,0.43,0.88]; %设置点的颜色,三个数字分别[R G B]的权重，基于0~1之间 [0.02,0.71,0.29];
g=gramm('x',X,'y',Y); %指定横轴x和纵轴y的取值，并创建gramm绘图对象
g.geom_point(); %绘制散点图
g.stat_glm(); %绘制依据散点图拟合的直线及置信区间
g.set_names('x','Funcational connectivity','y','Structural connectivity'); %设置坐标轴的标题
g.set_text_options('base_size' ,16,'label_scaling' ,1.2);%设置字体大小，基础字体大小base_size设为16，坐标轴的标题字体大小设为基础字体大小的1.2倍
g.set_point_options('base_size', 8,'step_size', 2);%设置散点大小
g.set_line_options('base_size', 3,'step_size', 2);%设置回归线的大小
g.set_color_options('map',color_point); %设置点的颜色
g.set_title('Coupling of SC_FC_HC r=0.1660 p=0.2873');%设置总的标题名称
g.draw(); %设置好以上属性后，开始绘制
hold on
stats = ScatterOutliers(HC_FC_data,HC_SC_data);  %抛点


