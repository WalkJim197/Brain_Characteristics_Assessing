% IC_number = [39,9,20,49,35,40,12,25,28,34,14,44,2,16,37,38,7,17,21,22,29,45];%GIG-ICA
% IC_number = [27,21,37,44,36,8,18,30,31,13,50,11,26,49,4,12,15,16];%IVA-GL
TC_number = [1,8,39,9,20,35,40,12,25,28,14,24,19,2,4,37,7,17,21,22,29,31,45];%GIG-ICA
% TC_number = [22,6,27,21,37,36,8,30,31,19,42,50,11,26,49,2,4,16,12,15];%IVA-GL
for i = 1 : 184
    tc = load(['E:\ASD\PostPrep\match components\TC_IVA\iva_ica_br',num2str(i),'.mat'],'tc');
    tc = getfield(tc,'tc')';
    
    for j = IC_number
        idx = find(IC_number == j);
        new_ic(idx,:) = tc(j,:);
    end
%     [m,n] = size(new_ic);%mutual information method
%     mu_fnc = [];
%     for x = 1 : m
%         for y = 1 : m
%             [~,I]=VectorMI(new_ic(x,:)', new_ic(y,:)');
%             mu_fnc(x,y) = I;
%         end
%     end
    fnc(:,:,i) = corr(new_ic');%pearson correlation
    At = fnc;
    m  = (1:size(At,1))' < (1:size(At,2));
    v1  = At(m);
    ALL_fnc(i,:) = mean(v1);
end
m=mean(fnc,3);
for k = 1:length(m)
m(k,k)=m(k,k)-1;
end

regress_var = xlsread('E:\ASD\cov.xlsx','sheet3');
% save('E:\ASD\PostPrep\variables\regress_var.mat','regress_var');
% GIG_AU_fnc = ALL_fnc([1:79],:);
% GIG_HC_fnc = ALL_fnc([80:end],:);
% AU_regress_var = regress_var([1:79],:);
% HC_regress_var = regress_var([80:end],:);
[GIG_AU_Mfnc_regress, b, stats] = regress_out(GIG_ALL_Mfnc([1:79],:), regress_var([1:79],:));
% save('E:\ASD\PostPrep\variables\GIG_ALL_fnc_regress.mat','GIG_ALL_fnc_regress');
goals_var = xlsread('E:\ASD\ASDnew.xlsx','goals');
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
miss_idx = find(goals_var(:,7) == -9999);
clinical = goals_var(:,7);
clinical(miss_idx) = [];%remove missing value
fnc = GIG_AU_fnc_regress;
fnc(miss_idx) = [];

[r,p] = corr(clinical,fnc);
%导入用于绘制图形的数据
X = clinical;
Y = fnc;  
stats = ScatterOutliers(X,Y);  %抛点

%I = VectorMI(ALL_fnc,clinical);互信息
% 
% % %%画散点图
% scatter(ALL_fnc,clinical);  
% hold on;
% 
% Y = clinical;  
% X = [ones(length(ALL_fnc), 1), ALL_fnc];
% b= regress(Y,X);
% x = ALL_fnc;
% y = X*b;
% plot(x,y)
% hold on;

%%gramm 绘图

h = figure;
set(h,'units','normalized','position',[0.1 0.1 0.4 0.6]); %设置绘图窗口的大小[0.1 0.1 0.4 0.6]
set(h,'color','w'); %设置绘图窗口的背景为白色
color_point=[0.18,0.43,0.88]; %设置点的颜色,三个数字分别[R G B]的权重，基于0~1之间 [0.02,0.71,0.29];
g=gramm('x',X,'y',Y); %指定横轴x和纵轴y的取值，并创建gramm绘图对象
g.geom_point(); %绘制散点图
g.stat_glm(); %绘制依据散点图拟合的直线及置信区间
g.set_names('x','ADI_R_VERBAL_TOTAL_BV','y','Funtional connectivity'); %设置坐标轴的标题
g.set_text_options('base_size' ,16,'label_scaling' ,1.2);%设置字体大小，基础字体大小base_size设为16，坐标轴的标题字体大小设为基础字体大小的1.2倍
g.set_point_options('base_size', 8,'step_size', 2);%设置散点大小
g.set_line_options('base_size', 3,'step_size', 2);%设置回归线的大小
g.set_color_options('map',color_point); %设置点的颜色
g.set_title('r=0.30 p=0.01');%设置总的标题名称
g.draw(); %设置好以上属性后，开始绘制


