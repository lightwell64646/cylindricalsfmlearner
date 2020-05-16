import matplotlib.pyplot as plt
from utils import parameter_count

def evaluate_saliency(training_harness, net_class, dataset, loss_function, saliency_function, Flags):
    net = training_harness(Flags, net_class, dataset, loss_function, saliency_function)
    if Flags.init_checkpoint_file != None:
        net.load_model(Flags.init_checkpoint_file)
    net.train(Flags.initial_steps)
    net.save_explicit("saliency_base")

    metrics = net.getPruneMetric()
    for i, m in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        plt.hist(m.numpy())
    plt.savefig("saliency histogram") 
    plt.show(block = False)
 
    print(float(net.eval(Flags.eval_steps)))

    plot_x = []
    pre_y = post_y = []
    for i in range(Flags.num_prunes):
        net2 = training_harness(Flags, net_class, dataset, loss_function, saliency_function)
        net2.clone_prune_state(net)
        net2.load_model("saliency_base")
        net2.prune(kill_fraction = Flags.prune_rate*(i+1))
        pre = net.eval(Flags.eval_steps)
        net2.train(Flags.repair_steps)
        post = net.eval(Flags.eval_steps)
        pre_y.append(pre)
        post_y.append(post)
        plot_x.append(parameter_count(net2.net))
    plt.scatter(plot_x, pre_y)
    plt.scatter(plot_x, post_y)
    plt.savefig("saliency efficiency") 
    plt.show()
    results_file = open(Flags.saliency_report_path, "w")
    results_file.write("x,pre,post\n")
    for x,pre,post in zip(plot_x,pre_y,post_y):
        results_file.write(str(x)+','+str(pre)+','+str(post)+'\n')
    results_file.close()


    