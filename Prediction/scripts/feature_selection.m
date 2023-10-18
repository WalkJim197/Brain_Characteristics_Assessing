% IC_number = [39,9,20,49,35,40,12,25,28,34,14,44,2,16,37,38,7,17,21,22,29,45];%GIG-ICA
IC_number = [27,21,37,44,36,8,18,30,31,13,50,11,26,49,4,12,15,16];%IVA-GL
% TC_number = [1,8,39,9,20,35,40,12,25,28,14,24,19,2,4,37,7,17,21,22,29,31,45];%GIG-ICA
% TC_number = [22,6,27,21,37,36,8,30,31,19,42,50,11,26,49,2,4,16,12,15];%IVA-GL

%-------convert ic/tc to 3D data including subject dimension
for i = 1 : 184
    ic = load(['E:\ASD\PostPrep\match components\IC_IVA\iva_ica_br',num2str(i),'.mat'],'ic');
    ic = getfield(ic,'ic')';
    ic_sub(:,:,i) = ic'; %ic need transposition
end
newtc_sub = permute(ic_sub,[3,2,1]);%reshape ICnumber * tc * sub to sub * tc * ICnumber

%feature selection in each components
r_thrsh = 0; p_thrsh = 0.01;
regress_var = xlsread('E:\ASD\cov.xlsx','sheet3');%behavior measures
for j = IC_number
    fprintf('\n component j # %3.0f',j);
    for k = 1 : size(newtc_sub,2)%dimension of the tc/ic value
       sub_info = newtc_sub(:,k,j);%extract k th tc/ic value across subjects in the jth component
       %considering age distribution is not a normal(log can sovle it),
       %Spearman has been used in correlation section
       [r_mat, p_mat] = corr(sub_info, regress_var(:,1), 'type', 'Spearman');
       if r_mat > r_thrsh && p_mat < p_thrsh
           feature(:,k,j) = sub_info;% select k th tc/ic value in j th component
           r(j,k) = r_mat;%save valid r and p value, component number (j) * tc/ic value (k)
           p(j,k) = p_mat;
       end
    end
end



