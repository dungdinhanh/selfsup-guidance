import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16.4})
plt.rcParams["figure.figsize"] = (7.1,5.2)
FOLDER="../runs_local/analyse/IMN64/conditional"

# scales=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
scales=[2.0, 6.0, 10.0]

# scales_signs = ["0p0", "2p0", "4p0", "6p0", "8p0", "10p0"]
scales_signs = ["2p0",  "6p0", "10p0"]

conflict_types = ["classification_diversity", "diffusion_classification", "diffusion_diversity"]

# colors = [u'#1f77b4', u'#ff7f0e', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
colors = [u'#1f77b4', u'#ff7f0e',    u'#7f7f7f', u'#bcbd22', u'#17becf'] # grey
# colors = [u'#1f77b4', u'#ff7f0e', u'#9467bd' ]


def gather_info(folder):
    n = len(scales)
    m = len(conflict_types)

    for j in range(m):
        list_coss = []
        list_mags = []
        save_folder = os.path.join(folder, conflict_types[j])
        os.makedirs(save_folder, exist_ok=True)
        for i in range(n):
            log_file = os.path.join(folder, f"scale{scales_signs[i]}", "reference" ,conflict_types[j], "coss_mags.npz")
            npz_info = np.load(log_file)
            coss_matrix = npz_info['arr_0']
            mags_matrix = npz_info['arr_1']
            list_coss.append(coss_matrix)
            list_mags.append(mags_matrix)
        plot_all(list_coss, list_mags, save_folder)

def plot_all(list_coss, list_mags, folder):
    n = len(list_mags)
    m = len(list_coss)
    list_coss_d = []
    list_mags_d = []
    assert m == n
    for i in range(n):
        distilled_coss, distill_mags =  distill_information(list_coss[i], list_mags[i])
        list_coss_d.append(distilled_coss)
        list_mags_d.append(distill_mags)
    x_axis = np.arange(list_coss[0].shape[0])
    for i in range(n):
        plt.plot(x_axis[:-1], list_coss_d[i][:-1], label=f"scale {scales[i]}", color=colors[i])
        # if i % 2 == 0 or i == (n-1) :
        #     plt.plot(x_axis, list_coss_d[i], label=f"scale {scales[i]}")
    # plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("percentage of conflict samples (%)")
    plt.legend()
    plt.savefig(os.path.join(folder, "cosine_sim.png"))
    plt.close()

    for i in range(n):
        plt.plot(x_axis, list_mags_d[i], label=f"scale {scales[i]}", color=colors[i])
    plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("average magnitude similarity")
    plt.savefig(os.path.join(folder, "magnitude_sim.png"))
    plt.close()

def plot_all_1fig(list_coss, list_mags, folder):
    n = len(list_mags)
    m = len(list_coss)
    list_coss_d = []
    list_mags_d = []
    assert m == n
    for i in range(n):
        distilled_coss, distill_mags =  distill_information(list_coss[i], list_mags[i])
        list_coss_d.append(distilled_coss)
        list_mags_d.append(distill_mags)
    x_axis = np.arange(list_coss[0].shape[0])
    for i in range(n):
        plt.plot(x_axis[:-1], list_coss_d[i][:-1], label=f"scale {scales[i]}", color=colors[i])
        # if i % 2 == 0 or i == (n-1) :
        #     plt.plot(x_axis, list_coss_d[i], label=f"scale {scales[i]}")
    # plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("percentage of conflict samples (%)")
    plt.legend()
    plt.savefig(os.path.join(folder, "cosine_sim.png"))
    plt.close()

    for i in range(n):
        plt.plot(x_axis, list_mags_d[i], label=f"scale {scales[i]}", color=colors[i])
    plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("average magnitude similarity")
    plt.savefig(os.path.join(folder, "magnitude_sim.png"))
    plt.close()

def distill_information(timestep_coss, timestep_mags):
    n = len(timestep_mags)
    m = len(timestep_coss)
    assert m == n
    distilled_coss = []  # number of negative /number of >=0
    distilled_mags = []  # average magnitudes
    for i in range(n):
        neg_coss_idx = timestep_coss[i] < 0.0
        neg_coss = timestep_coss[i][neg_coss_idx]
        neg_mags = timestep_mags[i][neg_coss_idx]

        no_samples = timestep_coss[i].shape[0]
        no_neg = neg_coss.shape[0]
        if no_neg == 0:
            avg_mags = 0
        else:
            avg_mags = np.mean(neg_mags)
        rate_neg = no_neg / no_samples * 100
        distilled_coss.append(rate_neg)
        distilled_mags.append(avg_mags)
    return distilled_coss, distilled_mags
    pass

if __name__ == '__main__':
    gather_info(FOLDER)