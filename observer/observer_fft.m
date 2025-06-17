function observer_fft(cost, options, output)
%
% observer0(cost, options, output)
%
% Print and display information on minimization algorithm such as the
% current estimate, the number of iterations, plot the cost functions
%
% Input:
%   cost    : is an array struct representing a cost function sum_k f_k(T_kx)
%   options : is a struct with options (algorithm, cost)
%   output  : is the current output of the minimization algorithm
%
% Nelly Pustelnik  (nelly.pustelnik@ens-lyon.fr)
% Laurent Condat   (laurent.condat@gipsa-lab.grenoble-inp.fr)
% Jerome Boulanger (jerome.boulanger@curie.fr)


k = output.iteration;
x1 = output.estimate;

if k == 0
    fprintf(1, 'Algorithm : %s\n', options.algorithm);
    for k = 1:size(cost,2)
        if isfield(cost(k).function, 'data')
            fprintf(1, '  Cost %d: %s (%s x, y)\n', k, cost(k).function.name, cost(k).operator.name);
        else
            fprintf(1, '  Cost %d: %s (%s x)\n', k, cost(k).function.name, cost(k).operator.name);
        end
    end
else

if mod(k, options.naff) == 0
    if isfield(output, 'cost')
        subplot(121)
    end
    if ndims(x1)==1
        plot(x1,'bo:')
    elseif ndims(x1) == 2
      subplot(121)
      imshow(x1,[])
      subplot(122)
      imshow(fftshift(log(abs(fft2(x1))+.1)),[]);
    elseif ndims(x1) == 3
      subplot(121)
      imshow(x1(:,:,1),[]);
      subplot(122)
      imshow(fftshift(log(abs(fft2(x1(:,:,1)))+.1)),[]);
    end
    axis tight;axis square;
    title(sprintf('iter:%d/%d', k, options.max_iter))
    if isfield(output, 'cost')
        subplot(122)
        colors = {'g','b','y','m'};
        cost_min = min(output.cost(:));
        for k = 1:length(cost)
            if (cost_min < 0)
                plot(output.cost(:,k), colors{k}); hold on
            else
                loglog(output.cost(:,k), colors{k}); hold on
            end
            L{k} = sprintf('%d: %s(%s)', k, ...
                cost(k).function.name, cost(k).operator.name);
        end
        if (cost_min < 0)
            plot(sum(output.cost, 2), 'r--', 'LineWidth', 2); hold on
        else
            loglog(sum(output.cost, 2), 'r--', 'LineWidth', 2); hold on
        end
        L{k+1} = 'Total';
        legend(L)
        hold off
        title(sprintf('%s',options.algorithm))
        ylabel('Cost functions')
        xlabel('Iterations')
        axis tight;axis square;
    end
    drawnow
end
end
