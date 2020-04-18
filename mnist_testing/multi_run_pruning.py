from utils import parameter_count
from copy import deepcopy
from time import time

def cultivate_model(training_harness, net_class, dataset, loss_function, Flags, prune_levels = [0.1,0.2,0.4]):
    print(Flags.initial_training_steps, Flags.prune_recovery_steps, Flags.prune_recovery_log_count, Flags.eval_steps)

    results_file = open(Flags.cultivation_report_path, "w")

    start_time = time()
    first_model = training_harness(Flags, net_class, dataset, loss_function)
    if Flags.init_checkpoint_file != None:
        first_model.load_model(Flags.init_checkpoint_file)
    first_model.train(Flags.initial_training_steps)
    eval_results = first_model.eval()

    first_save_path = Flags.checkpoint_dir + "first"
    first_model.save_explicit(first_save_path)
    results_file.write("Prune Cycle Begining\nParameterCount, Eval Time (s), Prune Level, Prune Breaks")
    for i in range(Flags.prune_recovery_log_count):
        results_file.write(",Accuracy(" + str(i) + ")")
    results_file.write("\n")
    results_file.write(str(float(eval_results)) + "," + str(time() - start_time) + "," + str(int(parameter_count(first_model.net))) + "\n")

    last_iteration_prune_metrics = first_model.getPruneMetric()
    last_iteration_save_path = first_save_path
    last_iteration_net = first_model

    for iter in range(Flags.max_prune_cycles):
        nets = [training_harness(Flags, net_class, dataset, loss_function) for _ in range(len(prune_levels))]
        best_save_path = None
        best_prune_metrics = None
        best_net = None
        best_prune_factor = 0
        best_perf = 0
        best_parameter_count = 0

        for net, prune in zip(nets, prune_levels):
            start_time = time()
            net.clone_prune_state(last_iteration_net)
            net.load_model(last_iteration_save_path)
            save_path = Flags.checkpoint_dir + "prune"+str(prune)+"_itr"+str(iter+1) + "/"
            print("last_iteration prune metric", [l.shape for l in last_iteration_prune_metrics])
            prune_breaks = [float(br) for br in net.prune(kill_fraction = prune)]
            net.save_explicit(save_path)
            param_count = parameter_count(net.net)

            results_file.write(str(prune) + "," + str(prune_breaks))
            iterations_per_step = Flags.prune_recovery_steps / Flags.prune_recovery_log_count
            for steps in range(Flags.prune_recovery_log_count):
                net.train(iterations_per_step)
                net_perf = net.eval(Flags.eval_steps)
                if steps == 0:
                    results_file.write(str(int(param_count)) + "," + str(time() - start_time))
                results_file.write("," + str(float(net_perf)))
                print("intermediate evaluation: ", net_perf)
            results_file.write("\n")
                
            adjusted_perf = net_perf - (int(param_count)/Flags.target_parameter_count)*Flags.parameter_value_weighting

            if best_save_path == None or adjusted_perf > best_perf:
                best_save_path = save_path
                best_prune_metrics = net.getPruneMetric()
                best_net = net
                best_prune_factor = prune
                best_perf = adjusted_perf
                best_parameter_count = param_count


        last_iteration_prune_metrics = best_prune_metrics
        last_iteration_save_path = best_save_path
        last_iteration_net = best_net
        results_file.write("finished training cycle best performance " + str(float(best_perf)) +
             " achieved when " + str(best_prune_factor*100) + "% of weights pruned\n")
             
        if best_parameter_count <= Flags.target_parameter_count:
            break

    results_file.close()