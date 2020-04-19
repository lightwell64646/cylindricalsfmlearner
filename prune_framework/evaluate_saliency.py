import matplotlib.pyplot as plt

def evaluate_saliency(training_harness, net_class, dataset, loss_function, Flags):
    net = training_harness(Flags, net_class, dataset, loss_function)
    if Flags.init_checkpoint_file != None:
        net.load_model(Flags.init_checkpoint_file)
    net.train(Flags.initial_steps)

    metrics = net.getPruneMetric()
    for i, m in enumerate(metrics):
        plt.subplot(len(metrics), 1, i+1)
        plt.hist(m.numpy())
    plt.show(block = False)

    print(float(net.eval(Flags.eval_steps)))

    

    '''prune_metrics = net.getPruneMetric()
    for i in range(Flags.num_prunes):
        net.prune(kill_fraction = Flags.prune_rate)
        print("performance after prune ", net.eval(Flags.eval_steps))
        net.train(Flags.repair_steps)
        print("performance after repair ", net.eval(Flags.eval_steps))'''


    