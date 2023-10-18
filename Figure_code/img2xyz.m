%%%%%%%%% step 1 %%%%%%%%%
%% load *.img and return the x,y,z coordinates
V = spm_vol('E:\ASD\PostPrep\new_mancovan\IVA\sm_stats\iva_mancovan_sm_HC - ASD_sig_effects_comp_008.img');%if input *.nii, please apply function "spm_vol_nifti.m"; *.img is "spm_vol"
[Y,XYZmm] = spm_read_vols(V);
%reshape whole brain size 61*73*61 to 1*271633
y = reshape(Y,1,[]);
%convert t-value matrix to p-value matrix 
% Let v be your degrees of freedom
v = 182;
p = 2*(1-tcdf(abs(y),v)); 
%search the significant p values and corresponding idex
p_sig = p(p<0.05);
p_sig_idx = find(p<0.05);
p_idxC = [p_sig_idx;p_sig];
p_xyz = XYZmm(:,p_sig_idx);
p_c = mean(p_sig);%cluster-level p value
[M,I] = max(abs(y));%where M is max t value,I is the index in whole brain
pp_xyz = XYZmm(:,I);%peak-level coordinate
pp = p(I);%peak-level p value
% %%%%%%%%% step 2 %%%%%%%%%
% %load the brodmann template
% br = spm_vol_nifti('D:\research_toolbox\dpabi\DPABI_V6.2_220915\Templates\brodmann.nii');
% [w,coor] = spm_read_vols(br);
% W = reshape(w,1,[]);
% %search the brain region based on brodmann (version 1.0)
% % p_W = [];
% % for i = 1 : length(p_xyz)
% %     for j = 1 : length(coor)
% %         L = p_xyz(:,i) == coor(:,j);
% %         L_sum = sum(L);
% %         if L_sum == 3
% %             u=length(p_W);
% %             p_W(1,u+1) = j;
% %         end
% %     end
% % end
% 
% %version 2.0
% re_p_xyz = [p_xyz(1,:)+91; p_xyz(2,:)+126; p_xyz(3,:)+72];%reorient the coordinantes (between brodmann and ours)
% p_W = [];
% for i = 1:length(re_p_xyz)
%     p_W(1,i) = w(re_p_xyz(1,i), re_p_xyz(2,i), re_p_xyz(3,i));
% end
% [unique_bramap0,~] = unique(p_W,'stable');
% unique_bramap = unique_bramap0(unique_bramap0 > 0)';

%load the aal template
Reference0 = load(['E:\ASD\PostPrep\aal_Labels.mat'],'Reference');
Reference = getfield(Reference0,'Reference');
aal = spm_vol_nifti('D:\research_toolbox\dpabi\DPABI_V6.2_220915\Templates\aal.nii');
[w,coor] = spm_read_vols(aal);
W = reshape(w,1,[]);

re_p_xyz = [p_xyz(1,:)+91; p_xyz(2,:)+126; p_xyz(3,:)+72];%reorient the coordinantes (between brodmann and ours)
p_W = [];
for i = 1:length(re_p_xyz)
    p_W(1,i) = w(re_p_xyz(1,i), re_p_xyz(2,i), re_p_xyz(3,i));
end
[unique_bramap0,~] = unique(p_W,'stable');
unique_bramap = unique_bramap0(unique_bramap0 > 0)';
for j = 1:length(unique_bramap)
    aal_name(j,1) = Reference(unique_bramap(j)+1);
end

%% draw tbe histograms

% change name of sub010 to sub10, like this. 
for i = 80:99
    HC = load_nii(['E:\ASD\PostPrep\stats\task1-1\IVA\13\3D\HC\iva_sub0',num2str(i),'_component_ica_s1__013.nii']);
    save_nii(HC,['E:\ASD\PostPrep\stats\task1-1\IVA\13\3D\HC\iva_sub',num2str(i),'_component_ica_s1__013.nii'])
    
end
%extract the magnitude of voxels in the peak coordinates for each group
group_mag_ASD = [];
for j = 1:79
    peak_c = spm_vol_nifti(['E:\ASD\PostPrep\stats\task1-1\GIG-ICA\12\3D\ASD\gig_sub',num2str(j),'_component_ica_s1__012.nii']);
    [V,C] = spm_read_vols(peak_c);
    v = reshape(V,1,[]);
    group_mag_ASD(1,j) = V(25+12, 62, 26);
end

% [h,p,ci] = ttest2(group_mag_HC,group_mag_ASD);

h1 = histogram(group_mag_HC);
hold on 
h2 = histogram(group_mag_ASD);
% h1.Normalization = 'probability';
h1.BinWidth = 0.25;
% h2.Normalization = 'probability';
h2.BinWidth = 0.25;

%% Test for difference in mean of variance maps
varMASK_HC = spm_vol_nifti('E:\ASD\PostPrep\stats\task3\IVA\16\varMASK_HC.nii');
[v1,c1] = spm_read_vols(varMASK_HC);
varMASK_ASD = spm_vol_nifti('E:\ASD\PostPrep\stats\task3\IVA\16\varMASK_ASD.nii');
[v2,c2] = spm_read_vols(varMASK_ASD);
V1 = reshape(v1,1,[]);
V2 = reshape(v2,1,[]);
V1_non0 = nonzeros(V1);
V2_non0 = nonzeros(V2);
[p,h,stats] = ranksum(V1_non0,V2_non0);

%draw the scatter using the normalized value after -log10(p values)

iva = [1.18831246745955;2.18605411723914;0.0768847952159342;
    2.82970864795231;0.103274805488126;2.63867983678228;2.77549283527292;
    8.58502665202918;68.3233063903751;3.03011835625350;11.3882766919927;1.02259741580629]';

gig = [0.589141115130238;1.81553803946827;33.9956786262174;
    1.13062854796005;0.962405980374246;4.03011835625350;1.84875849553242;
    3.32697909287110;4.99567862621736;10.6903698325741;17.1051303432547;0.0471203538599321]';
labels = [1,2,3,4,5,6,7,8,9,10,11,12];

scatter(labels,iva,500,'s','filled')
hold on 
scatter(labels,gig,500,'s','filled')
hold on
plot([0,13],[1.301,1.301],'--')
set(gca, 'Box', 'on' ,'TickDir', 'in')
hold on 
scatter(8,50)

%% Test for voxelwise difference of variance between groups (voxelwise F test):

V = spm_vol_nifti('E:\ASD\PostPrep\stats\task4\GIG-ICA\9\F2_spm\F.nii');
[Y,XYZmm] = spm_read_vols(V);
[Data_Corrected, Header, P]=y_FDR_Image(Y,qThreshold,OutputName,MaskFile,Flag,Df1,Df2,VoxelSize,Header)














