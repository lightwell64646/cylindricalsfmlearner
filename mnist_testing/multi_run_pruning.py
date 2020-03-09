from mnist_net import parameter_count
from copy import deepcopy

def cultivate_model(net_class, Flags, prune_levels = [0.05,0.1,0.2,0.4], iterations = 3):
    results_file = open(Flags.cultivation_report_path, "w")

    first_model = net_class(Flags)
    first_model.train()
    eval_results = first_model.eval()

    first_save_path = Flags.checkpoint_dir + "/fract0_itr0"
    first_model.save_explicit(first_save_path)
    results_file.write("Prune Cycle Begining\nAccuracy,ParameterCount\n")
    results_file.write(str(eval_results) + "," + str(parameter_count(first_model.mnist_net)) + "\n")

    last_iteration_prune_metrics = first_model.getPruneMetric()
    last_iteration_save_path = first_save_path

    for iter in range(iterations):
        nets = [net_class(Flags) for _ in range(len(prune_levels))]
        best_save_path = None
        best_prune_metrics = None
        best_prune_factor = 0
        best_perf = 0
        performance = []
        for net, prune in zip(nets, prune_levels):
            net.load_mnist_model(last_iteration_save_path)
            net.setPruneMetric(deepcopy(last_iteration_prune_metrics))
            save_path = Flags.checkpoint_dir + "fract"+str(prune)+"_itr"+str(iter+1)
            print("last_iteration prune metric", [l.shape for l in last_iteration_prune_metrics])
            net.prune(kill_fraction = prune)
            net.train()

            net_perf = net.eval()
            performance.append(net_perf)
            if best_save_path == None or net_perf > best_perf:
                best_save_path = save_path
                best_prune_metrics = net.getPruneMetric()
                best_prune_factor = prune
                best_perf = net_perf

            results_file.write(str(net_perf) + "," + str(parameter_count(net.mnist_net)) +"\n")

        last_iteration_prune_metrics = best_prune_metrics
        last_iteration_save_path = best_save_path
        results_file.write("finished training cycle best performance " + str(best_perf) +
             " achieved when " + str(best_prune_factor) + " of weights pruned\n")