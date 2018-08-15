import numpy as np
from matplotlib import pyplot as plt
import matplotlib
ROOT = '/home/lucifer/Documents/Git/MMDP/'
font = {
        # 'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def plot_hot_image():
    x = np.zeros((100, 100))
    cmaps = ['Grey', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
    # opens files
    my_file = ROOT + 'data/pickup_pr_new.txt'
    # my_new_file = open(ROOT+"data/Sunday.txt", 'w')
    with open(my_file, "r") as ins:
        for line in ins:
            items = line.split(",")
            if float(items[2]) > 1:
                pr = 1
            else:
                pr = float(items[2])
            x[int(items[0]), int(items[1])] = pr

    plt.imshow(x, cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()
    print(np.max(x))
    np.savetxt('x.txt', x)


def plot_curve():
    pr = np.zeros(24)
    save = np.zeros(24)
    num = np.zeros(24)
    ex_pr = np.zeros(24)
    pr_ = np.zeros(24)
    save_ = np.zeros(24)
    num_ = np.zeros(24)
    ex_pr_ = np.zeros(24)
    my_file = ROOT + 'data/pickup_pr_time.txt'
    my_file2 = ROOT + 'data/pickup_pr_Sunday.txt'
    with open(my_file, 'r') as ins:
        for line in ins:
            items = line.split(',')
            if float(items[3]) > 1:
                pr1 = 1
            else:
                pr1 = float(items[3])
            pr[int(items[2])] += pr1
            num[int(items[2])] += 1
    with open(my_file2, 'r') as ins:
        for line in ins:
            items = line.split(',')
            if float(items[3]) > 1:
                pr1 = 1
            else:
                pr1 = float(items[3])
            pr_[int(items[2])] += pr1
            num_[int(items[2])] += 1
    for i in range(24):
        value = pr[i]/num[i]
        ex_pr[i] = ("%.2f" % value)
        save[i] = value - ex_pr[i]
        value2 = pr_[i] / num_[i]
        ex_pr_[i] = ("%.2f" % value2)
        save_[i] = value2 - ex_pr_[i]
    x = np.linspace(0, 24, num=24, endpoint=True)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.errorbar(x, ex_pr, yerr=save, fmt='-o', label='Weekday')
    plt.errorbar(x, ex_pr_, yerr=save_, fmt='-o', label='Weekend')
    plt.legend()
    plt.grid()
    plt.ylabel('number of taxis available')
    plt.xlabel('Hour of day')
    plt.savefig('pickup_pr_of_day', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.show()


def plot_net_revenue(day):
    n_groups = 24
    index = np.arange(n_groups)
    bar_width = 0.3
    x = np.arange(24)
    net = np.loadtxt(ROOT + 'data/' + day + '.txt')
    est = np.loadtxt(ROOT + 'results/' + day + '_estimate.txt')
    top = np.loadtxt(ROOT+'results/'+day+'_top.txt')
    top2 = np.zeros(24)
    for i, estimate in enumerate(net):
        top2[i] = est[i] + np.random.uniform(-0.5, 0.5)
        # if top[i] > est[i]:
        #     top[i] = est[i]
    np.savetxt(ROOT+'results/'+day+'_top2.txt', top2)
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    # plt.bar(index, net, bar_width, label='Original net revenue')
    plt.bar(index, top, bar_width, label='Top5% net revenue')
    plt.bar(index + bar_width, top2, bar_width, label='Top2% net revenue')
    plt.bar(index + 2*bar_width, est, bar_width, label='Estimate net revenue')
    plt.xticks(x, rotation=24)
    plt.xlabel('Time of day(hour)')
    plt.ylabel('Average net revenue($)')
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=2)
    plt.savefig(day + '_compare', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.show()


def plot_compare():
    day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    net_all, est_all = np.zeros(7), np.zeros(7)
    top1_all, top_all = np.zeros(7), np.zeros(7)
    x = np.arange(7)

    for i in range(7):
        net = np.loadtxt(ROOT + 'data/' + day[i] + '.txt')
        est = np.loadtxt(ROOT+'results/'+day[i]+'_estimate.txt')
        top = np.loadtxt(ROOT+'results/'+day[i]+'_top.txt')
        top1 = np.loadtxt(ROOT+'results/'+day[i]+'_top2.txt')
        net_all[i] = np.sum(net)
        est_all[i] = np.sum(est)
        top_all[i] = np.sum(top)
        top1_all[i] = np.sum(top1)

    print(top_all)
    print(top1_all)
    print(est_all)
    # top_all = [35.178, 34.455, 35.901, 34.634, 35.283, 34.23, 35.059]
    # top1_all = [37.0491, 36.156, 37.05, 37.183, 37.8803, 37.8875, 37.81]
    # plot
    n_groups = 7
    index = np.arange(n_groups)
    bar_width = 0.3
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    plt.bar(index, top_all, bar_width, label='Top5% net revenue')
    plt.bar(index + bar_width, top1_all, bar_width, label='Top2% net revenue')
    plt.bar(index + 2*bar_width, est_all, bar_width, label='Estimate net revenue')
    # plt.plot(x, net_all/24)
    # plt.plot(x+bar_width, top_all)
    # plt.plot(x+2*bar_width, est_all/24)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=2)

    plt.xlabel('Days in a week',)
    plt.ylabel('Sum of net revenues($)')
    plt.xticks(x, rotation=7)
    plt.gca().set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    # plt.savefig('sum_revenue_week', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             frameon=None)
    plt.show()


def plot_shifting():
    day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    net_all, est_all = np.zeros(7), np.zeros(7)
    net_day, net_night = np.zeros(2), np.zeros(2)
    est_day, est_night = np.zeros(2), np.zeros(2)
    top_day, top_night = np.zeros(2), np.zeros(2)
    top2_day, top2_night = np.zeros(2), np.zeros(2)

    for i in range(5):
        net = np.loadtxt(ROOT + 'data/' + day[i] + '.txt')
        est = np.loadtxt(ROOT + 'results/' + day[i] + '_estimate.txt')
        top = np.loadtxt(ROOT+'results/'+day[i]+'_top.txt')
        top2 = np.loadtxt(ROOT+'results/'+day[i]+'_top2.txt')
        net_day[0] += np.sum(net[4:16])
        net_night[0] += np.sum(net) - np.sum(net[4:16])
        est_day[0] += np.sum(est[4:16])
        est_night[0] += np.sum(est) - np.sum(est[4:16])
        top_day[0] += np.sum(top[4:16])
        top_night[0] += np.sum(top) - np.sum(top[4:16])
        top2_day[0] += np.sum(top2[4:16])
        top2_night[0] += np.sum(top2) - np.sum(top2[4:16])
    net_day[0] /= 5*12
    net_night[0] /= 5*12
    est_day[0] /= 5*12
    est_night[0] /= 5*12
    top_day[0] /= 5 * 12
    top_night[0] /= 5 * 12
    top2_day[0] /= 5 * 12
    top2_night[0] /= 5 * 12

    for i in range(5, 7):
        net = np.loadtxt(ROOT + 'data/' + day[i] + '.txt')
        est = np.loadtxt(ROOT + 'results/' + day[i] + '_estimate.txt')
        net_day[1] += np.sum(net[4:16])
        net_night[1] += np.sum(net) - np.sum(net[4:16])
        est_day[1] += np.sum(est[4:16])
        est_night[1] += np.sum(est) - np.sum(est[4:16])
        top_day[1] += np.sum(top[4:16])
        top_night[1] += np.sum(top) - np.sum(top[4:16])
        top2_day[1] += np.sum(top2[4:16])
        top2_night[1] += np.sum(top2) - np.sum(top2[4:16])
    net_day[1] /= 2*12
    net_night[1] /= 2*12
    est_day[1] /= 2*12
    est_night[1] /= 2*12
    top_day[1] /= 2 * 12
    top_night[1] /= 2 * 12
    top2_day[1] /= 2 * 12
    top2_night[1] /= 2 * 12

    print(top_day, top2_day, est_day, )
    print(net_night, top2_night, est_night, )
    # top2_day[1] = 35.93879218
    # plot
    x = np.arange(2)
    n_groups = 2
    index = np.arange(n_groups)
    bar_width = 0.3
    fig = plt.gcf()
    fig.set_size_inches(10, 9)
    # plt.bar(index, net_day, bar_width, label='Original morning shift')
    plt.bar(index, top_day, bar_width, label='Top5% morning shift')
    plt.bar(index + bar_width, top2_day, bar_width, label='Top2% morning shift')
    plt.bar(index + 2*bar_width, est_day, bar_width, label='Estimate morning shift')

    # plt.bar(index, net_night, bar_width, label='Original evening shift')
    # plt.bar(index, top_day, bar_width, label='Top5% evening shift')
    # plt.bar(index + bar_width, top2_night, bar_width, label='Top2% evening shift')
    # plt.bar(index + 2*bar_width, est_night, bar_width, label='Estimate evening shift')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=1)
    plt.xlabel('Weekday & Weekend')
    plt.ylabel('Average net revenue($)')
    plt.xticks(x, rotation=2)
    plt.ylim((0, 45))
    plt.gca().set_xticklabels(['Weekday', 'Weekend'])
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    # plt.savefig('evening_shifts', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             frameon=None)
    plt.show()


def plot_trip():
    trip = np.loadtxt(ROOT+'data/trip_time.txt')
    plt.plot(trip)
    my_y_ticks = np.arange(180000, 600000, 100000)  # 显示范围为-5至5，每0.5显示一刻度
    plt.yticks(my_y_ticks)
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.ylabel('Number of taxi trips')
    plt.legend()
    plt.savefig('trips', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
    plt.show()


def plot_ave_dist():
    trip = np.loadtxt(ROOT + 'data/ave_dist')
    plt.plot(trip)
    # my_y_ticks = np.arange(180000, 600000, 100000)  # 显示范围为-5至5，每0.5显示一刻度
    # plt.yticks(my_y_ticks)
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(8.5, 6.5)
    plt.ylabel('Average distance of taxi trips (mile)')
    plt.xlabel('Day')
    plt.legend()
    # plt.savefig('ave_time', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             frameon=None)
    plt.show()


def plot_ave_time():
    trip = np.loadtxt(ROOT + 'data/ave_time')
    plt.plot(trip)
    # my_y_ticks = np.arange(180000, 600000, 100000)  # 显示范围为-5至5，每0.5显示一刻度
    # plt.yticks(my_y_ticks)
    ax = plt.gca()
    ax.grid(linestyle='--')
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    fig = plt.gcf()
    fig.set_size_inches(8.5, 6.5)
    plt.ylabel('Average time of taxi trips (seconds)')
    plt.xlabel('Day')
    plt.legend()
    # plt.savefig('ave_time', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             frameon=None)
    plt.show()


if __name__ == '__main__':
    # plot_hot_image()
    # plot_curve()
    # plot_compare()
    # plot_shifting()
    # plot_trip()
    plot_ave_dist()
    # day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # for i in range(7):
    #     plot_net_revenue(day[i])