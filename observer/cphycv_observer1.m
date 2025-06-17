function cphycv_observer1(k, cost, options, output, x0, x1)
if k == 0
  fprintf(1,'Starting cphycv\n  rho:%g\n  tau:%g\n  sigma:%g\n  max iter:%d\n', ...
	     options.rho, options.tau, options.sigma, options.max_iter);
else
  if mod(k, options.naff) == 0
    figure(1)
    subplot(121)
    imshow(cat(3,cat(3,x1{1},x1{2}),zeros(size(x1{1}))),[])
    title(sprintf('iter:%d/%d', k, options.max_iter))
    subplot(122)
    colors = {'g','b','y','m'};
    for k = 1:size(cost,2)
        semilogy(output.cost(:,k),colors{k}); hold on
        L{k} = sprintf('Cost %d', k);
    end
    semilogy(sum(output.cost,2),'r--','LineWidth',2); hold on
    L{k+1} = 'Total';
    legend(L)
    hold off
    title('Cost')
    axis tight;axis square;
    drawnow
  end
end
