import matplotlib.pyplot as plt
import numpy as np
from utils import parameter_count
import tensorflow as tf

def evaluate_saliency(training_harness, net_class, dataset, loss_function, saliency_function, Flags, run_name):
    net = training_harness(Flags, net_class, dataset, loss_function, saliency_function)
    if Flags.init_checkpoint_file != None:
        net.load_model(Flags.init_checkpoint_file)
    initial_logs = log_training(net, Flags.initial_steps, Flags)

    #net.save_explicit(run_name + "_saliency_base")

    metrics = net.getPruneMetric() 
    
    plt.figure(run_name + " Saliency Distribution")
    plt.title(run_name +" Saliency Distribution" )
    plt.xlabel(run_name + " Saliency")
    plt.ylabel("Neurons")
    sqrMet = int(np.ceil(np.sqrt(len(metrics))))
    for i, m in enumerate(metrics):
        plt.subplot(sqrMet, sqrMet, i+1)
        plt.xlabel(run_name + " Saliency")
        plt.ylabel("Neurons")
        plt.hist(m.numpy())
    plt.savefig(run_name + " Saliency histogram")
    if (Flags.show_plots):
        plt.show(block = False)
 
    print(initial_logs[-1])

    plot_x = []
    logs = []
    for i in range(Flags.num_prunes):
        #print(net.net.layers[0].variables[0][0,0,0])
        net.prune(kill_fraction = Flags.prune_rate)
        breaking = False
        for s in net.saliency:
            if (tf.reduce_prod(s.shape) == 0):
                print("pruned to 0 resolving")
                breaking = True
                break
        if breaking:
            break
        #print(net.net.layers[0].variables[0][0,0,0])
        if (Flags.repair_steps != 0):
            new_logs = log_training(net, Flags.repair_steps, Flags)
            logs.append(new_logs)
        else:
            logs.append([net.eval(Flags.eval_steps)])
        plot_x.append(float(parameter_count(net.net)))
    print(logs, initial_logs)
    plt.figure(run_name + " Prune Effect")
    
    plt.title(run_name + " Prune Accuracy" + " No Repairs" if Flags.repair_steps == 0 else "")
    plt.xlabel("Total Parameter Count")
    total_eval_size = Flags.batch_size * Flags.eval_steps if (Flags.eval_steps is not None) else "Full Test"
    plt.ylabel("Accuracy (Batch=" + str(total_eval_size) + ")")
    for i in range(len(logs[0])):
        plt.scatter(plot_x, [l[i] for l in logs])
    plt.savefig(run_name + " Saliency Efficiency") 
    if (Flags.show_plots):
        plt.show(block = False)
    results_file = open(Flags.saliency_report_path + " " + run_name + ".csv", "w")
    results_file.write("x,Pre ... Post Prune\n")
    
    results_file.write("initial, ")
    for l in initial_logs:
        results_file.write(str(l)+',')
    results_file.write('\n')
    
    for x,log in zip(plot_x, logs):
        results_file.write(str(x)+',')
        for l in log:
            results_file.write(str(l)+',')
        results_file.write('\n')
    results_file.close()

def log_training(net, steps, Flags):
    log = [net.eval(Flags.eval_steps)]
    for _ in range(Flags.training_log_resolution):
        net.train(steps/Flags.training_log_resolution)
        log.append(net.eval(Flags.eval_steps))
    return log
    