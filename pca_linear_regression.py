import torch
import torchvision
import videotransforms
import os
import glob
import time
import pickle

import numpy as np

from i3d_model import InceptionI3d
from torch import nn
from torchvision import transforms
from pyramid_pooling import SpatialPyramidPooling, TemporalPyramidPooling
from i3d_train import set_one_thread
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from perform_encoding import get_fmri, predict_fmri_fast, vectorized_correlation, saveasnii
from nilearn import plotting


end_points = ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5b',
              'Mixed_5c', 'Logits']
ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']
subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

def save_pca_activations():
    mode = "rgb"
    work_dir = "./i3d_dir/pca_activations/rgb_imagenet"
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if mode == 'flow':
        fp = "/Users/gautham/src/pytorch-i3d/models/flow_imagenet.pt"
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(fp))
    else:
        fp = "/Users/gautham/src/pytorch-i3d/models/rgb_imagenet.pt"
        # fp = "/Users/gautham/src/pytorch-i3d/models/rgb_charades.pt"
        # fp = "/home/vasan/scratch/pytorch-i3d/models/rgb_charades.pt"
        i3d = InceptionI3d(400, in_channels=3, final_endpoint="Logits")
        i3d.load_state_dict(torch.load(fp))
    i3d.to(device)

    data_transform = transforms.Compose([videotransforms.CenterCrop(224)])

    all_vids = glob.glob('./participants_data/AlgonautsVideos268_All_30fpsmax/*.mp4')
    all_vids.sort()
    train_vids = all_vids[:1000]
    test_vids = all_vids[1000:]

    tp_pool = TemporalPyramidPooling(levels=[1, 2, 4, 8, 16], mode='max')
    sp_pool = SpatialPyramidPooling(levels=[2, 4, 8], mode='max')
    avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))

    pca_data = {}
    for ep in end_points:
        pca_data[ep] = []

    for i_vid, video_fp in enumerate(train_vids):
        tic = time.time()
        video_frames, audio_frames, metadata = torchvision.io.read_video(filename=video_fp)
        video_frames = data_transform(video_frames)
        video_frames = torch.permute(video_frames, (3, 0, 1, 2))  # c, t, h, w
        video_frames = torch.unsqueeze(video_frames, dim=0)
        total_frames = video_frames.shape[2]
        indices = np.linspace(0, total_frames - 1, 60, dtype=np.int)
        video_frames = ((video_frames[:, :, indices, :, :] / 255.) * 2) - 1

        # pred = i3d(video_frames)
        # phi = torch.mean(pred, axis=-1)
        last_layer_features, all_phi = i3d.extract_features(video_frames.to(device))

        for ep in end_points:
            if ep in ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3']:
                feat = tp_pool(all_phi[ep][0]) # [64, 930]
                feat = torch.mean(feat, 0)
            elif ep == 'MaxPool3d_4a_3x3':
                feat = sp_pool(all_phi[ep][0]) # [480, 105]
                feat = torch.mean(feat, 0)
            elif ep == "Logits":
                feat = last_layer_features.view((-1, 7))
                feat = torch.mean(feat, 1)
            else:
                feat = avg_pool(all_phi[ep])
                feat = feat.view((-1, 7))
                feat = torch.mean(feat, 1)
            pca_data[ep].append(feat.detach().cpu().ravel())
        print(i_vid, "{:.3f}".format(time.time() - tic))

    for ep in end_points:
        x = pca_data[ep]
        x = torch.cat(x).view((len(x), -1))
        x = StandardScaler().fit_transform(x)
        ipca = PCA(n_components=100, random_state=42)
        ipca.fit(x)
        x = ipca.transform(x)
        save_path = os.path.join(work_dir, "{}".format(ep))
        np.save(save_path, x)

def train_pca_linear_regression():
    batch_size = 1000
    roi_comparison_data = {}
    # dataset = "rgb_imagenet"
    dataset = "rgb_charades"
    for ep in end_points:
        activations = np.load("./i3d_dir/pca_activations/{}/{}.npy".format(dataset, ep))
        plot_data = np.zeros((len(ROIs), len(subjects)))
        for i_roi, roi in enumerate(ROIs):
            for subject in subjects:
                corr_scores = []
                for i in range(10): # Cross validation
                    fmri_dir = "./participants_data/participants_data_v2021/mini_track/sub{}".format(subject)
                    fmri_data = get_fmri(fmri_dir, roi)
                    num_voxels = fmri_data.shape[1]
                    inds = np.arange(1000)
                    np.random.shuffle(inds)

                    fmri_train = fmri_data[inds[:900], :]
                    fmri_test = fmri_data[inds[900:], :]

                    train_activations = activations[inds[:900], :]
                    test_activations = activations[inds[900:], :]
                    pred_fmri = np.zeros_like(fmri_test)

                    # print("number of voxels is ", num_voxels)
                    iter = 0
                    while iter < num_voxels - batch_size:
                        pred_fmri[:, iter:iter + batch_size] = predict_fmri_fast(train_activations, test_activations,
                                                                                 fmri_train[:, iter:iter + batch_size],
                                                                                 use_gpu=False)
                        iter = iter + batch_size
                        print((100 * iter) // num_voxels, " percent complete")
                    pred_fmri[:, iter:] = predict_fmri_fast(train_activations, test_activations,
                                                            fmri_train[:, iter:iter + batch_size], use_gpu=False)

                    score = vectorized_correlation(fmri_test, pred_fmri)
                    print("{}: Mean correlation for ROI: {} in subject {} is: {}".format(
                        ep, roi, subject, np.mean(score)))
                    corr_scores.append(np.mean(score))
            plot_data[i_roi, :] = np.array(corr_scores)

        roi_comparison_data[ep] = plot_data.copy()

    with open("./i3d_dir/pca_activations/{}/all_subjects_data.pkl".format(dataset), "wb") as handle:
        pickle.dump(roi_comparison_data, handle)

def train_wb_pca():
    batch_size = 1000
    roi_comparison_data = {}
    # dataset = "rgb_imagenet"
    dataset = "rgb_charades"
    activations = []

    for ep in end_points:
    # for ep in ['MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5c', 'Logits']:
        a = np.load("./i3d_dir/pca_activations/{}/{}.npy".format(dataset, ep))
        activations.append(a.copy())
    activations = np.concatenate(activations, axis=1)

    for subject in subjects:
        corr_scores = []
        for i in range(10):  # Cross validation
            fmri_dir = "./participants_data/participants_data_v2021/full_track/sub{}".format(subject)
            fmri_data, voxel_mask = get_fmri(fmri_dir, 'WB')
            num_voxels = fmri_data.shape[1]
            inds = np.arange(1000)
            np.random.shuffle(inds)

            fmri_train = fmri_data[inds[:900], :]
            fmri_test = fmri_data[inds[900:], :]

            train_activations = activations[inds[:900], :]
            test_activations = activations[inds[900:], :]
            pred_fmri = np.zeros_like(fmri_test)

            # print("number of voxels is ", num_voxels)
            iter = 0
            while iter < num_voxels - batch_size:
                pred_fmri[:, iter:iter + batch_size] = predict_fmri_fast(train_activations, test_activations,
                                                                         fmri_train[:, iter:iter + batch_size],
                                                                         use_gpu=False)
                iter = iter + batch_size
                print((100 * iter) // num_voxels, " percent complete")
            pred_fmri[:, iter:] = predict_fmri_fast(train_activations, test_activations,
                                                    fmri_train[:, iter:iter + batch_size], use_gpu=False)

            score = vectorized_correlation(fmri_test, pred_fmri)
            print("Mean correlation for ROI: {} in subject {} is: {}".format('WB', subject, np.mean(score)))
            corr_scores.append(np.mean(score))
        print('-' * 100)
        print("Mean validation score for sub{}: {}".format(subject, np.mean(corr_scores)))
        print('-' * 100)


def visualize():
    fmri_dir = "./participants_data/participants_data_v2021/full_track/sub{}".format(subject)
    fmri_data, voxel_mask = get_fmri(fmri_dir, 'WB')

    visual_mask_3D = np.zeros((78, 93, 71))
    visual_mask_3D[voxel_mask == 1] = score
    brain_mask = './example.nii'
    nii_save_path = os.path.join(results_dir, ROI + '_val.nii')
    saveasnii(brain_mask, nii_save_path, visual_mask_3D)
    view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage', \
                                     title='Correlation for sub' + sub, colorbar=False)
    view_save_path = os.path.join(results_dir, ROI + '_val.html')
    view.save_as_html(view_save_path)
    print("Results saved in this directory: ", results_dir)
    view.open_in_browser()


if __name__ == '__main__':
    set_one_thread()    # Handle torch shenanigans
    tic = time.time()
    # save_activations()
    # save_pca_activations()
    # train_pca_linear_regression()
    train_wb_pca()
    print("Activation generation took: {:.3f}".format(time.time() - tic))
