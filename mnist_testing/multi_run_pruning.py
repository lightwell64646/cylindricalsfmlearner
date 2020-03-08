from copy import deepcopy

def cultivate_model(net_class, Flags, prune_levels = [0.05,0.1,0.2,0.4], iterations = 3):
    results_file = open(Flags.cultivation_report_path, "w")

    first_model = net_class(Flags)
    first_model.train()
    eval_results = first_model.eval()

    first_save_path = Flags.checkpoint_dir + "firstTrainedCkpt"
    first_model.save_explicit(first_save_path)
    results_file.write(
    results_file.write(str(eval_results) + "\n")

    last_iteration_prune_metrics = first_model.getPruneMetric()
    last_iteration_save_path = first_save_path

    for iter in iterations:
        nets = [net_class(Flags) for _ in range(prune_levels)]
        best_save_path = None
        best_prune_metrics = None
        best_prune_factor = 0
        best_perf = 0
        performance = []
        for net, prune in zip(nets, prune_levels):
            net.load_mnist_model(last_iteration_save_path)
            net.setPruneMetric(last_iteration_prune_metrics)
            save_path = Flags.checkpoint_dir + "pune/fraction"+str(prune)+"_iter"+str(iter)
            net.prune(kill_fraction = prune, 
                save_path=save_path)
            net.train()

            net_perf = net.eval()
            performance.append(net_perf)
            if best_save_path == None or net_perf > best_perf:
                best_save_path = save_path
                best_prune_metrics = net.getPruneMetric()
                best_prune_factor = prune
                best_perf = net_perf

        last_iteration_prune_metrics = best_prune_metrics
        last_iteration_save_path = best_save_path
        results_file.write("eval," + str(best_perf) + "\n")