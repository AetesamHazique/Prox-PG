function img = load_stack(filename)
% Load a tiff stack as a 3D array
stack = tiffread32(filename);
img = zeros(size(stack(1).data,1),size(stack(2).data,2),length(stack));
for i = 1:length(stack)
    img(:,:,i) = stack(i).data;
end
