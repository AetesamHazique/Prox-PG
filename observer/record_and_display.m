function output = record_and_display(cost, options, output, x, t0)

output.estimate = x;

if isfield(output, 'iteration')
    output.iteration = output.iteration + 1;
else
    output.iteration = 1;
end

if options.record == 1
    for i = 1:length(cost)
        output.cost(output.iteration,i) = cost(i).eval(x);
    end
    output.elapsed_time(output.iteration) = toc(t0);
else
    output.elapsed_time = toc(t0);
end

if isfield(options, 'observer')
    options.observer(cost, options, output);
end
