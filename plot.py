import matplotlib
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.summary.summary_iterator import summary_iterator
from beautifultable import BeautifulTable
from sys import platform
if platform == "darwin":
    # matplotlib.use("MacOSX")
    matplotlib.use("TKAgg")


layer_ind_to_activation = {
    '1': 'MaxPool3d_2a_3x3',
    '2': 'MaxPool3d_3a_3x3',
    '3': 'MaxPool3d_4a_3x3',
    '4': 'MaxPool3d_5a_2x2',
    '5': 'Mixed_5b',
    '6': 'Mixed_5c',
    '7': 'Logits',
}

input_dims = {
    '1': 930,
    '2': 930,
    '3': 1260,
    '4': 832,
    '5': 832,
    '6': 1024,
    '7': 1024,
}

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
ROIs = ['V1', 'V2', 'V3', 'V4', 'PPA', 'STS', 'LOC', 'FFA', 'EBA']
colors =['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
end_points = ['Logits', 'MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c']


def bar_plots():
    results_dir = "./i3d_dir/rgb_charades"
    plot_dir = "./i3d_dir/rgb_charades/plots"
    # results_dir = "./i3d_dir/rgb_imagenet"
    # plot_dir = "./i3d_dir/rgb_imagenet/plots"
    layers_inds = [str(x) for x in range(1, 8)]

    roi_comparison_data = {}
    for l_ind in layers_inds:
        layer_name = layer_ind_to_activation[l_ind]
        table = BeautifulTable(maxwidth=140, precision=6)
        table.set_style(BeautifulTable.STYLE_MARKDOWN)
        # plt.Figure()
        plot_data = np.zeros((len(ROIs), len(subjects)))
        for i_roi, roi in enumerate(ROIs):
            corr_scores = []
            for i_sub, subject in enumerate(subjects):
                rpath = "/{}/sub{}/{}".format(layer_name, subject, roi)
                fpath = results_dir + rpath
                fpath = glob.glob(fpath + "/events*")
                assert len(fpath) == 1
                fpath = fpath[0]
                data = np.zeros((10, 30))
                counter = 0
                for summary in summary_iterator(fpath):
                    if summary.summary.value:
                        if summary.summary.value[0].tag == "Correlation/val":
                            x = counter // 30
                            y = counter % 30
                            val = summary.summary.value[0].simple_value
                            data[x, y] = val
                            counter += 1
                assert counter == 300
                corr = np.mean(np.max(data, 1))
                corr_scores.append(corr)
            table.rows.append(corr_scores)
            plot_data[i_roi, :] = np.array(corr_scores)
        table.columns.header = subjects
        table.rows.header = ROIs
        print(layer_name)
        print(table)

        # Subject wise plots
        # for i_sub in range(len(subjects)):
        #     fig = plt.figure()
        #     plt.bar(ROIs, plot_data[:, i_sub], color=colors)
        #     plt.xlabel("ROIs", fontsize=14, fontweight="bold")
        #     plt.ylim([0., 0.4])
        #     h = plt.ylabel("Mean\nCorrelation", fontsize=14, fontweight="bold", labelpad=50)
        #     h.set_rotation(0)
        #     # plt.title("Layer: {}, Subject: {}".format(layer_name, i_sub), fontsize=16, fontweight="bold")
        #     plt.tight_layout()
        #     save_path = plot_dir + "/sub{}/{}.png".format(subjects[i_sub], layer_name)
        #     plt.savefig(save_path)
        #     plt.show()

        # Avg correlation over 10 subjects
        fig = plt.figure()
        plt.bar(ROIs, np.mean(plot_data, 1), color=colors)
        plt.xlabel("ROIs", fontsize=14, fontweight="bold")
        plt.ylim([0., 0.4])
        h = plt.ylabel("Mean\nCorrelation", fontsize=14, fontweight="bold", labelpad=50)
        h.set_rotation(0)
        save_path = plot_dir + "/all_subjects-{}.png".format(layer_name)
        plt.title("{}".format(layer_name), fontsize=16, fontweight="bold")
        plt.tight_layout()
        roi_comparison_data[layer_name] = plot_data.copy()
        # plt.savefig(save_path)
        # plt.show()

    with open(plot_dir + "/all_subjects_data.pkl", "wb") as handle:
        pickle.dump(roi_comparison_data, handle)

def comparison_bar_plots():
    imagenet_data = pickle.load(open("./i3d_dir/rgb_imagenet/plots/all_subjects_data.pkl", "rb"))
    charades_data = pickle.load(open("./i3d_dir/rgb_charades/plots/all_subjects_data.pkl", "rb"))
    pca_charades_data = pickle.load(open("./i3d_dir/pca_activations/rgb_charades/all_subjects_data.pkl", "rb"))
    pca_imagenet_data = pickle.load(open("./i3d_dir/pca_activations/rgb_imagenet/all_subjects_data.pkl", "rb"))

    # all_data = [pca_imagenet_data, pca_charades_data]
    all_data = [imagenet_data, charades_data]
    # all_data = [imagenet_data, charades_data, pca_imagenet_data, pca_charades_data]
    index = np.arange(len(ROIs))
    bar_width = 0.2

    # labels = ["PCA-Kinetics", "PCA-Charades"]
    labels = ["NN-Kinetics", "NN-Charades"]
    c = colors[:2]
    # c = colors[2:4]

    plot_dir = "./i3d_dir/comparison_plots"
    for ep in end_points:
        fig = plt.figure()
        for i_col, data in enumerate(all_data):
            x = np.mean(data[ep], 1)
            plt.bar(index + (i_col * bar_width), x, width=bar_width, color=c[i_col], label=labels[i_col])
        plt.xlabel("ROIs", fontsize=14, fontweight="bold")
        plt.xticks(index, ROIs)
        plt.ylim([0., 0.4])
        h = plt.ylabel("Mean\nCorrelation", fontsize=14, fontweight="bold", labelpad=50)
        h.set_rotation(0)
        # save_path = plot_dir + "/all_subjects-{}.png".format(layer_name)
        plt.title("{}".format(ep), fontsize=16, fontweight="bold")
        plt.legend()
        plt.tight_layout()
        save_path = plot_dir + "/nn-all_subjects-{}.png".format(ep)
        plt.savefig(save_path)
        # plt.show()

if __name__ == '__main__':
    # bar_plots()
    comparison_bar_plots()