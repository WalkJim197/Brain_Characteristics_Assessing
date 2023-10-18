clc
clear all

% FC_HIP_R_TP = xlsread('F:\project\SFC\SFC_paper_submit\response2comments\FC_HIP_R_TP_99.xlsx');
% save FC_HIP_R_TP;
% mean_MD_R_hipp_TP = xlsread('F:\project\SFC\SFC_paper_submit\response2comments\mean_MD_R_hipp_TP_99.xlsx');
% save mean_MD_R_hipp_TP;
% regress_var_99 = xlsread('F:\project\SFC\�˿���Ϣͳ�Ʊ�\99_age_sex_edu.xlsx');
% save regress_var_99;

%%׼������
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\���ݱ��ع��˵�\mean_FC_HIP_R_TP_cor.mat');
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\���ݱ��ع��˵�\mean_MD_HIP_R_TP_cor.mat');
mean_FC_HIP_R_TP_cor = r_mean_FC_HIP_R_TP;
mean_MD_HIP_R_TP_cor = r_mean_MD_HIP_R_TP;

HC_FC_data = mean_FC_HIP_R_TP_cor(1:43);
VMCI_FC_data = mean_FC_HIP_R_TP_cor(44:77);
MCI_FC_data = mean_FC_HIP_R_TP_cor(78:99);

HC_SC_data = mean_MD_HIP_R_TP_cor(1:43);
VMCI_SC_data = mean_MD_HIP_R_TP_cor(44:77);
MCI_SC_data = mean_MD_HIP_R_TP_cor(78:99);

load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\����δ�ع�\mean_FC_HIP_R_TP.mat');
load('F:\project\SFC\SFC_paper_submit\response2comments\coupling_SC_FC\����δ�ع�\mean_MD_HIP_R_TP.mat');


HC_FC_data = mean_FC_HIP_R_TP(1:43);
VMCI_FC_data = mean_FC_HIP_R_TP(44:77);
MCI_FC_data = mean_FC_HIP_R_TP(78:99);

HC_SC_data = mean_MD_HIP_R_TP(1:43);
VMCI_SC_data = mean_MD_HIP_R_TP(44:77);
MCI_SC_data = mean_MD_HIP_R_TP(78:99);

%%������أ��õ�б�ʺ�Pֵ
[r,p] = corr(HC_FC_data,HC_SC_data);
% figure;

% %%��ɢ��ͼ
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

%%gramm ��ͼ
%�������ڻ���ͼ�ε�����
X = HC_FC_data;
Y = HC_SC_data;  
stats = ScatterOutliers(X,Y);  %�׵�

h=figure;
set(h,'units','normalized','position',[0.1 0.1 0.4 0.6]); %���û�ͼ���ڵĴ�С[0.1 0.1 0.4 0.6]
set(h,'color','w'); %���û�ͼ���ڵı���Ϊ��ɫ
color_point=[0.18,0.43,0.88]; %���õ����ɫ,�������ֱַ�[R G B]��Ȩ�أ�����0~1֮�� [0.02,0.71,0.29];
g=gramm('x',X,'y',Y); %ָ������x������y��ȡֵ��������gramm��ͼ����
g.geom_point(); %����ɢ��ͼ
g.stat_glm(); %��������ɢ��ͼ��ϵ�ֱ�߼���������
g.set_names('x','Funcational connectivity','y','Structural connectivity'); %����������ı���
g.set_text_options('base_size' ,16,'label_scaling' ,1.2);%���������С�����������Сbase_size��Ϊ16��������ı��������С��Ϊ���������С��1.2��
g.set_point_options('base_size', 8,'step_size', 2);%����ɢ���С
g.set_line_options('base_size', 3,'step_size', 2);%���ûع��ߵĴ�С
g.set_color_options('map',color_point); %���õ����ɫ
g.set_title('Coupling of SC_FC_HC r=0.1660 p=0.2873');%�����ܵı�������
g.draw(); %���ú��������Ժ󣬿�ʼ����
hold on
stats = ScatterOutliers(HC_FC_data,HC_SC_data);  %�׵�


