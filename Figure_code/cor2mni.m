%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mni = cor2mni(cor, T)
% function mni = cor2mni(cor, T)
% convert matrix coordinate to mni coordinate
%
% cor: an Nx3 matrix
% T: rotation matrix
% mni is the returned coordinate in mni space
%
% caution: if T is not given, the default T is
% T = [-3,0,0,93;0,3,0,-129;0,0,3,-75;0,0,0,1];
%if nargin == 1
 %   T = [-3,0,0,93;0,3,0,-129;0,0,3,-75;0,0,0,1];
%end

cor = round(cor);
mni = T*[cor(:,1) cor(:,2) cor(:,3) ones(size(cor,1),1)]';
mni = mni';
mni(:,4) = [];
return;
end