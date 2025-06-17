function cphycv_observer0(k, cost, options, output, x0, x1)

if k == 0
  fprintf(1,'Starting cphycv\n  rho:%g\n  tau:%g\n  sigma:%g\n  max iter:%d\n', ...
	     options.rho, options.tau, options.sigma, options.max_iter);
     for k = 1:size(cost,2)
         if isfield(cost(k).function, 'data') 
            fprintf(1, 'Cost %d: %s (%s x, y)\n', k, cost(k).function.name, cost(k).operator.name);
         else
            fprintf(1, 'Cost %d: %s (%s x)\n', k, cost(k).function.name, cost(k).operator.name);
         end
     end
else
  if mod(k, options.naff) == 0
    figure(1)
    subplot(121)
    imshow(x1,[])
    title(sprintf('iter:%d/%d', k, options.max_iter))
    subplot(122)
    colors = {'g','b','y','m'};
    % compute the minium of the costs to know if it is positive
    cost_min = 1e-12;
    for k = 1:length(cost)
        cost_min = min(cost_min, output.cost(:,k));
    end
    for k = 1:length(cost)
        if (cost_min < 0)
            plot(output.cost(:,k), colors{k}); hold on
        else
            semilogy(output.cost(:,k), colors{k}); hold on
        end
        L{k} = sprintf('Cost %d: %s(%s)', k, cost(k).function.name, cost(k).operator.name);
    end
    if (cost_min < 0)
        plot(sum(output.cost, 2), 'r--', 'LineWidth', 2); hold on
    else
        semilogy(sum(output.cost, 2), 'r--', 'LineWidth', 2); hold on
    end
    L{k+1} = 'Total';
    legend(L)
    hold off
    title('Cost')
    axis tight;axis square;
    drawnow
  end
end
