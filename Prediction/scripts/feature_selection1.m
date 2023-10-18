% IC_number = [39,9,20,49,35,40,12,25,28,34,14,44,2,16,37,38,7,17,21,22,29,45];%GIG-ICA
IC_number = [27,21,37,44,36,8,18,30,31,13,50,11,26,49,4,12,15,16];%IVA-GL
% TC_number = [1,8,39,9,20,35,40,12,25,28,14,24,19,2,4,37,7,17,21,22,29,31,45];%GIG-ICA
% TC_number = [22,6,27,21,37,36,8,30,31,19,42,50,11,26,49,2,4,16,12,15];%IVA-GL
% mask = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1\IVA\26\mean_HCASD_MASK.nii']);

%-------convert ic/tc to 3D data including subject dimension
for i = 1 : 184
    ic = load(['E:/ASD/PostPrep/match components/IC_IVA/iva_ica_br',num2str(i),'.mat'],'ic');
    ic = getfield(ic,'ic')';
    ic_sub(:,:,i) = ic'; %ic need transposition
end
newtc_sub = permute(ic_sub,[3,2,1]);%reshape ICnumber * tc * sub to sub * tc * ICnumber

%feature selection in each components
r_thrsh = 0; p_thrsh = 0.05; %0.001
regress_var = xlsread('E:/ASD/PostPrep/prediction/Newfeatures/cov.xlsx','Sheet3');%behavior measures
% for j = IC_number
for j = 26
    fprintf('\n component j # %3.0f',j);
    for k = 1 : size(newtc_sub,2)%dimension of the tc/ic value
       sub_info = newtc_sub(:,k,j);%extract k th tc/ic value across subjects in the jth component
       %considering age distribution is not a normal(log can sovle it),
       %Spearman has been used in correlation section
%        [r_mat, p_mat] = corr(sub_info, regress_var(:,1), 'type', 'Spearman');
       %other metrics are object to normal distribution
       [r_mat, p_mat] = corr(sub_info, regress_var(:,1));
       if abs(r_mat) > r_thrsh && p_mat < p_thrsh
           feature(:,k,j) = sub_info;% select k th tc/ic value in j th component
           r(j,k) = r_mat;%save valid r and p value, component number (j) * tc/ic value (k)
           p(j,k) = p_mat;
       end
    end
end

% [~,~,cov]=xlsread('E:\ASD\PostPrep\variables\SHENGMIN\ic_edit.xlsx','Sheet4');
% label = cell2mat(cov(1:18,1));   
% net = cov(1:18,2);   
% 
% for i =1:length(label) 
%     [row,col,v] = find(feature(:,:,label(i))>0);
%     b = unique(col);
%     ech_comp = feature(:,:,label(i));
%     flag = 0;
%     for n = 1 : size(ech_comp,2)
%         if any(ech_comp(:,n)) == 1
%             flag = flag + 1;
%             pred_feature(:,flag) = ech_comp(:,n);
%         end
%     end
%     name = cell2mat(strcat('Age_',net(i)));
%     filename = ['E:\ASD\PostPrep\prediction\Newfeatures\pred_features\IVA_IC2\Age\',name,'.mat'];
%     save(filename,'pred_feature');
%     clear pred_feature
% end

%feature visualization
[row,col,v] = find(feature(:,:,26) ~= 0);
b = unique(col);
peak_c = spm_vol_nifti(['E:\ASD\PostPrep\GIG-ICA\gigMask.nii']);
[V,C] = spm_read_vols(peak_c);
vv = reshape(V,1,[]);
[row1,col1,v1] = find(vv>0);
for m = 1:length(b)
    z = b(m);
    b1(m) = col1(z);
end
cc = zeros(1,271633);
for mm = b1
    cc(mm) = 1;
end
f_c = reshape(cc,61,73,61);
origin = [31 43 25];voxel_size = [3 3 3];
f_nii = make_nii(f_c,voxel_size,origin);
save_nii(f_nii,['E:\ASD\PostPrep\prediction\Newfeatures\pred_features\GIG_IC\FIQ\FIQ_CRN.nii'])
flip_lr('E:\ASD\PostPrep\prediction\Newfeatures\pred_features\GIG_IC\FIQ\FIQ_CRN.nii',...
'E:\ASD\PostPrep\prediction\Newfeatures\pred_features\GIG_IC\FIQ\FIQ_CRN.nii');%Flip the L/R

