IC_number = [39,9,20,49,35,40,12,34,25,28,14,44,2,16,37,38,7,17,45,22,21,29];%GIG-ICA
% IC_number = [27,21,37,44,36,8,18,30,31,13,50,11,26,49,4,12,15,16];%IVA-GL

for g = IC_number
    clear feature
    % mask = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1\IVA\',num2str(g),'\mean_HCASD_MASK.nii']);
    mask = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1\GIG-ICA\',num2str(g),'\mean_HCASD_MASK.nii']);%edit
    [V1,C1] = spm_read_vols(mask);
    for i = 1 : 184
        sub = spm_vol_nifti(['E:\ASD\PostPrep\prediction\precessed data\gig\gig_sub',num2str(i),'_component_ica_s1_.nii'],g);%edit
        [V2,C2] = spm_read_vols(sub);
        v1 = reshape(V1,1,[]);
        v2 = reshape(V2,1,[]);
        [roww1,coll1,vv1] = find(v1~=0);
        [roww2,coll2,vv2] = find(v2~=0);
        newc = v1 .* v2;
        sub_newc(i,:) = newc;
    end

    r_thrsh = 0; p_thrsh = 0.1; 
    regress_var = xlsread('E:/ASD/PostPrep/prediction/Newfeatures/cov.xlsx','Sheet3');%behavior measures
    for m = 1 : size(sub_newc,2)
        sub_info = sub_newc(:,m);
        [r_mat, p_mat] = corr(sub_info, regress_var(:,6));%edit
           if abs(r_mat) > r_thrsh && p_mat < p_thrsh
               feature(:,m) = sub_info;
               r(1,m) = r_mat;
               p(1,m) = p_mat;
           end
    end
    save(['E:\ASD\PostPrep\prediction\precessed data\feature\comp\gig_com\PIQ\comp',num2str(g)],'feature');
end

[~,~,cov]=xlsread('E:\ASD\PostPrep\variables\SHENGMIN\ic_edit.xlsx','Sheet3');
label = cell2mat(cov(1:22,1));   
net = cov(1:22,2);   

for i =1:length(label) 
    comp = label(i);
    load(['E:\ASD\PostPrep\prediction\precessed data\feature\comp\gig_com\PIQ\comp',num2str(comp),'.mat'],'feature');
    flag = 0;
    for n = 1 : size(feature,2)
        if any(feature(:,n)) == 1
            flag = flag + 1;
            pred_feature(:,flag) = feature(:,n);
        end
    end
    name = cell2mat(strcat('PIQ_',net(i)));
    filename = ['E:\ASD\PostPrep\prediction\precessed data\feature\GIG-ICA\PIQ\',name,'.mat'];
    save(filename,'pred_feature');
    clear pred_feature feature
end


