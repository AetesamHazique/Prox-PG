function operator = create_identity_op(val)
%
% operator = create_identity_op(val)
%
% Output
%  operator :  struct with field as function 'apply', 'apply_adjoint' and real 'norm'
%

if nargin < 1
    val = 1;
end

operator.name = 'I';
operator.apply = @(x) val*x;
operator.apply_adjoint = @(x) val*x;
operator.norm = 1;
