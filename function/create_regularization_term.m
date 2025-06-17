function cost = create_regularization_term(lambda, type, img)

if strcmpi(type,'tikhonov')
    cost = create_tikhonov_cost(lambda);
elseif strcmpi(type,'tv')
    cost = create_total_variation_cost(lambda);
elseif strcmpi(type,'nltv')
    cost = create_non_local_total_variation_cost(lambda, img, 2, 2, (1+max(img(:)))/2);
elseif strcmpi(type,'hessian_schatten')
    cost = create_hessian_schatten_norm_cost(lambda);
elseif strcmpi(type,'patch_schatten')
    cost = create_patch_schatten_norm_cost(lambda, 4, 8);
else 
    error('unknown regularization (%s)', type)
end
    
