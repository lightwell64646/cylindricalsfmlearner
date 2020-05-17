import matplotlib.pyplot as plt
import numpy as np
from utils import parameter_count

def evaluate_saliency(training_harness, net_class, dataset, loss_function, saliency_function, Flags, run_name):
    net = training_harness(Flags, net_class, dataset, loss_function, saliency_function)
    if Flags.init_checkpoint_file != None:
        net.load_model(Flags.init_checkpoint_file)
    net.train(Flags.initial_steps)
    net.save_explicit(Flags.checkpoint_dir + run_name + "saliency_base")

    metrics = net.getPruneMetric()
    
    plt.figure(run_name + "saliency Dist")
    sqrMet = int(np.ceil(np.sqrt(len(metrics))))
    for i, m in enumerate(metrics):
        plt.subplot(sqrMet, sqrMet, i+1)
        plt.hist(m.numpy())
    plt.savefig(run_name + "saliency histogram")
    if (Flags.show_plots):
        plt.show(block = False)
 
    print(float(net.eval(Flags.eval_steps)))

    plot_x = []
    pre_y = []
    post_y = []
    for i in range(Flags.num_prunes):
        #net.load_model(Flags.checkpoint_dir + run_name + "saliency_base")
        #net.reset_prune_state(net)
        net.prune(kill_fraction = Flags.prune_rate)
        pre = net.eval(Flags.eval_steps)
        net.train(Flags.repair_steps)
        post = net.eval(Flags.eval_steps)
        pre_y.append(pre)
        post_y.append(post)
        plot_x.append(float(parameter_count(net.net)))
        print("intrim peek", pre, post, len(plot_x), len(pre_y), len(post_y))
    plt.figure(run_name + "prune effect")
    plt.scatter(plot_x, pre_y)
    plt.scatter(plot_x, post_y)
    plt.savefig(run_name + "saliency efficiency") 
    if (Flags.show_plots):
        plt.show(block = False)
    results_file = open(Flags.saliency_report_path + run_name, "w")
    results_file.write("x,pre,post\n")
    for x,pre,post in zip(plot_x,pre_y,post_y):
        results_file.write(str(x)+','+str(pre)+','+str(post)+'\n')
    results_file.close()


    