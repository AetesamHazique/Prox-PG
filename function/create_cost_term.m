function cost_term = create_cost_term(func, op)
%
% cost_term = create_cost_term(func,op)
%
% Create a cost term of the form 'func(op)'
%
% Input:
%  func : a struct with field the functions 'eval', 'prox'
%  op   : (optional) a struct with field the functions 'apply' and
%         'apply_adjoint' the scalar 'norm'.
%         If not provided the identity is assumed for T
%
% Output:
%  cost_term : a struct with fields 'function' and 'operator' representing
%  a term of a cost function of the type f(Tx)
%
% Example of use:
%  F.data = data;
%  F.eval = @(x) norm(x-data)^2 / 2;
%  F.prox = @(x,gamma) fun.data + (x - fun.data) / (1 + 2 * gamma * fun.lambda);
%  A.apply = @(x) imfilter(x,fspecial('gaussian'));
%  A.apply_adjoint = @(x) imfilter(x,fspecial('gaussian'));
%  A.norm = 1;
%  cost = create_cost_term(F, A);
%
% Nelly Pustelnik  (nelly.pustelnik@ens-lyon.fr)
% Laurent Condat   (laurent.condat@gipsa-lab.grenoble-inp.fr)
% Jerome Boulanger (jerome.boulanger@curie.fr)

if nargin < 2
    op = create_identity_op();
end

cost_term.function = func;
cost_term.operator = op;
cost_term.eval = @(x) cost_term.function.eval(cost_term.operator.apply(x));
