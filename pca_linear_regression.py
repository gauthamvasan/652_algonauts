import torch
import torchvision
import videotransforms
import os
import glob
import time
import numpy as np

from i3d_model import InceptionI3d
from torch import nn
from torchvision import transforms
from pyramid_pooling import SpatialPyramidPooling, TemporalPyramidPooling
from i3d_train import set_one_thread
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def save_pca_activations():
    mode = "rgb"
    work_dir = "./i3d_dir/pca_activations"
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if mode == 'flow':
        fp = "/Users/gautham/src/pytorch-i3d/models/flow_imagenet.pt"
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(fp))
    else:
        # fp = "/Users/gautham/src/pytorch-i3d/models/rgb_imagenet.pt"
        fp = "/Users/gautham/src/pytorch-i3d/models/rgb_charades.pt"
        # fp = "/home/vasan/scratch/pytorch-i3d/models/rgb_charades.pt"
        i3d = InceptionI3d(157, in_channels=3, final_endpoint="Logits")
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
    end_points = ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5b',
                  'Mixed_5c', 'Logits']
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
            elif ep == 'MaxPool3d_4a_3x3':
                feat = sp_pool(all_phi[ep][0]) # [480, 105]
            elif ep == "Logits":
                feat = last_layer_features.view((-1, 7))
            else:
                feat = avg_pool(all_phi[ep])
                feat = feat.view((-1, 7))
            pca_data[ep].append(feat.cpu().ravel())
        print(i_vid, "{:.3f}".format(time.time() - tic))

    for ep in end_points:
        x = pca_data[ep]
        x = torch.cat(x).view((len(x), -1)).detach()
        x = StandardScaler().fit_transform(x)
        ipca = PCA(n_components=100, random_state=42)
        ipca.fit(x)
        x = ipca.transform(x)
        save_path = os.path.join(work_dir, "{}".format(ep))
        np.save(save_path, x)

def train_pca_linear_regression():
    pass

if __name__ == '__main__':
    set_one_thread()    # Handle torch shenanigans
    tic = time.time()
    # save_activations()
    save_pca_activations()
    print("Activation generation took: {:.3f}".format(time.time() - tic))