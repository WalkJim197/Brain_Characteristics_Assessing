% IC_number = [39,9,20,49,35,40,12,25,28,34,14,44,2,16,37,38,7,17,21,22,29,45];%GIG-ICA
% IC_number = [27,21,37,44,36,8,18,30,31,13,50,11,26,49,4,12,15,16];%IVA-GL

% for k = 1 : 9 %Get rid of the extra zeros in file name
%     b=load_nii(['E:\ASD\PostPrep\prediction\precessed data\gig\gig_sub00',num2str(k),'_component_ica_s1_.nii']);
%     save_nii(b,['E:\ASD\PostPrep\prediction\precessed data\gig\gig_sub',num2str(k),'_component_ica_s1_.nii']);
% end
% mask = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1\IVA\\mean_HCASD_MASK.nii']);
mask = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1\GIG-ICA\17\mean_HCASD_MASK.nii']);
[V1,C1] = spm_read_vols(mask);
for i = 1 : 184
    sub = spm_vol_nifti(['E:\ASD\PostPrep\prediction\precessed data\gig\gig_sub',num2str(i),'_component_ica_s1_.nii'],17);
    [V2,C2] = spm_read_vols(sub);
    v1 = reshape(V1,1,[]);
    v2 = reshape(V2,1,[]);
    [roww1,coll1,vv1] = find(v1~=0);
    [roww2,coll2,vv2] = find(v2~=0);
    newc = v1 .* v2;
    sub_newc(i,:) = newc;
end
% save(['E:\ASD\PostPrep\prediction\precessed data\gig_com\comp39'],'sub_newc');


r_thrsh = 0; p_thrsh = 0.01; %0.001
regress_var = xlsread('E:/ASD/PostPrep/prediction/Newfeatures/cov.xlsx','Sheet3');%behavior measures
for m = 1 : size(sub_newc,2)
    sub_info = sub_newc(:,m);
    [r_mat, p_mat] = corr(sub_info, regress_var(:,1));
       if abs(r_mat) > r_thrsh && p_mat < p_thrsh
           feature(:,m) = sub_info;
           r(1,m) = r_mat;
           p(1,m) = p_mat;
       end
end

flag = 0;
for n = 1 : size(feature,2)
    if any(feature(:,n)) == 1
        flag = flag + 1;
        pred_feature(:,flag) = feature(:,n);
    end
end

%feature visualization
feature(184,271633) = 0;
fm = mean(feature,1);
ori_cor = reshape(fm,61,73,61);
origin = [31 43 25];voxel_size = [3 3 3];
ori_nii = make_nii(ori_cor,voxel_size,origin);
save_nii(ori_nii,['E:\ASD\PostPrep\prediction\precessed data\feature\GIG-ICA\FIQ_CRN.nii']) 
flip_lr('E:\ASD\PostPrep\prediction\precessed data\feature\GIG-ICA\FIQ_CRN.nii',...
'E:\ASD\PostPrep\prediction\precessed data\feature\GIG-ICA\FIQ_CRN.nii');%Flip the L/R

