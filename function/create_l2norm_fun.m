function fun = create_l2norm_fun(lambda, data)
%
% term = create_l2norm_fun(data)
%
% Create a cost function term of the form lambda * |.-data|^2/2 to be part of a
% problem solved by an optimization procedure
%
% Input:
%  lambda : multiplier
%  data   : data (optional)
% Output:
%  fun  : struct with field 'eval' and 'prox'
%
%  fun.eval = @(x) lambda norm( x - data )^2 /2
%  fun.prox = @(x,gamma) (x + lamdba * gamma * data) / (1 + lambda * gamma)
%
% Nelly Pustelnik  (nelly.pustelnik@ens-lyon.fr)
% Laurent Condat   (laurent.condat@gipsa-lab.grenoble-inp.fr)
% Jerome Boulanger (jerome.boulanger@curie.fr)

if nargin < 1
    lambda = 1;
end

fun.lambda = lambda;
fun.name = 'l2 norm';

if nargin > 1
  fun.data = data;
  fun.eval = @(x) fun.lambda * norm(x(:) - fun.data(:))^2 / 2;
  fun.prox = @(x, gamma) (x + gamma * fun.lambda * fun.data) / (1 + gamma * fun.lambda);
else
  fun.eval = @(x) fun.lambda * norm(x(:))^2 / 2;
  fun.prox = @(x, gamma) x / (1 + gamma * fun.lambda);
end
