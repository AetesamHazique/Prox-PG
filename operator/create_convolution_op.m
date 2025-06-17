function operator = create_convolution_op(A,domain,is_otf,dim)
%
% operator = create_convolution_op(A,domain,is_otf,dim)
%
% Input
%   A       : convolution filter mask (as produced by fspecial)
%   domaine : spectral / spatial domain (want to do the operation in spatial domain/ Fourier domain)
%   is_otf  : bool telling if A the otf when 'domain' is spectral
%   dim     : dimensions to compute the OTF from PSF
%
% Output
%  operator :  struct with field as function 'apply', 'apply_adjoint' and real 'norm'
%

if nargin < 2
    domain = 'spatial';
end

if nargin < 3
    is_otf = false;
end

if nargin < 4
    dim = size(A);
end

operator.name = 'H';
operator.domain = domain;
if ~strcmp(domain,'spectral')
    operator.filter = A;
    operator.adjoint_filter = A(end:-1:1,end:-1:1,end:-1:1); % mirror
    operator.norm = sum(A(:));
else
    if is_otf == true % Fourier domain
        operator.filter = A;
    else
        operator.filter = psf2otf(A, dim);
    end
    operator.adjoint_filter = conj(operator.filter);
    operator.resolvant = @(x,gamma) (fftn(x) ./ (1 + gamma * operator.adjoint_filter .* operator.filter));
    operator.norm = max(operator.filter(:));
end

operator.apply = @(x) apply_filter(operator.filter, x, operator.domain);
operator.apply_adjoint = @(x) apply_filter(operator.adjoint_filter, x, operator.domain);
