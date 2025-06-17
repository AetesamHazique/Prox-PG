%%
clear
close all
clc
addpath('../utils','../data','../solutions');

% Use this file as well: YRCPIR-EX9796-Cyan.tif
psffile = 'data/YRCPIR-EX9796-Yellow-PSF.tif'; 
% Use this file as well: YRCPIR-EX9796-Cyan-PSF
img=double(imread('data/YRCPIR-EX9796-Yellow.tif'));
% load and preprocess the point spread function
psf = load_stack(psffile);
psf = preprocess_point_spread_function(psf, size(psf),'sym');

%% Using My Proximal Technique

% Create the data fidelity term
T1 = create_convolution_op(psf);%,'spectral',false,size(data)); %A = create_composed_op(O);
cost(1) = create_cost_term(create_l2norm_fun(1,img1), T1);

lambda1 = 0.05;
T2 = create_identity_op(); %A = create_composed_op(O);
cost(2) = create_cost_term(create_l2norm_fun(lambda1,img1), T2);


% Create the regularization term
lambda2 = 0.5; % regularization parameter
cost(3) = create_regularization_term(lambda2,'tv', img1);

fprintf('lanmda1=%f,lambda2=%f',lambda1,lambda2);
% minimize the cost function
opts.max_iter = 100;
opts.tau = 2;
opts.observer = @observer0;
opts.record = 1;
output = cphycv(cost, opts);
