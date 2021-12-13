import argparse
import torch
import torchvision
import videotransforms
import os
import glob
import pickle
import time
import numpy as np

from i3d_model import InceptionI3d
from perform_encoding import get_fmri
from torch.utils.data import Dataset
from torch import nn
from mlp_models import SimpleMLP
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from utils.ols import vectorized_correlation
from tensorboardX import SummaryWriter
from pyramid_pooling import SpatialPyramidPooling, TemporalPyramidPooling

ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']

def set_one_thread():
  '''
  N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
  Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
  '''
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  torch.set_num_threads(1)

class I3DAlgonautsDataSet(Dataset):
    END_POINTS = ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5b',
                  'Mixed_5c', 'Logits']
    def __init__(self, activations_dir, fmri_dir, ROI, i3d_endpoint="", transform=None, target_transform=None, train=True):
        """

        Args:
            activations_dir:
            fmri_dir: './participants_data/participants_data_v2021'
            ROI:
            transform:
            target_transform:
        """
        assert i3d_endpoint in self.END_POINTS
        self.i3d_endpoint = i3d_endpoint
        self.activations_pkl = glob.glob(activations_dir + "/*.pkl")
        self.activations_pkl.sort()
        assert len(self.activations_pkl) != 0, "activations_dir cannot be empty"
        self.fmri_data = get_fmri(fmri_dir, ROI)
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.activations_pkl = self.activations_pkl[:900]
            self.fmri_data = self.fmri_data[:900, :]
        else:
            self.activations_pkl = self.activations_pkl[900:1000]
            self.fmri_data = self.fmri_data[900:, :]

    def __getitem__(self, item):
        fp = self.activations_pkl[item]
        activations = pickle.load(open(fp, "rb"))[self.i3d_endpoint]
        voxels = self.fmri_data[item]
        voxels = torch.from_numpy(voxels.astype(np.float32))
        return activations, voxels

    def __len__(self):
        return len(self.fmri_data)

def cross_validation_train():
    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('--sub', help='subject number from which real fMRI data will be used', default='01', type=str)
    parser.add_argument('--roi', help='brain region, from which real fMRI data will be used', default='LOC', type=str)
    parser.add_argument('--work_dir', help='Store results in this dir', default='./i3d_dir', type=str)
    parser.add_argument('--layer_ind', help='Pick a value in [1, 7]', default=1, type=int)
    args = parser.parse_args()

    # Configuration options
    ROI = args.roi
    subject = args.sub
    k_folds = 10
    num_epochs = 100
    hidden_sizes = [4096, 4096]
    batch_size_train = 16
    batch_size_test = 100
    loss_function = nn.MSELoss()
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
    work_dir = args.work_dir + "/{}/sub{}/{}".format(layer_ind_to_activation[str(args.layer_ind)], subject, ROI)
    
    # For fold results
    writer = SummaryWriter(work_dir)

    # Set fixed random number seed
    torch.manual_seed(42)

    # base_path = "/Users/gautham/src/652_algonauts"
    base_path = "/home/vasan/scratch/652_algonauts"
    train_dataset = I3DAlgonautsDataSet(
        fmri_dir=base_path + "/participants_data/participants_data_v2021/mini_track/sub{}".format(subject),
        ROI=ROI, activations_dir=base_path + "/i3d_dir/activations", train=True,
        i3d_endpoint=layer_ind_to_activation[str(args.layer_ind)]
    )

    test_dataset = I3DAlgonautsDataSet(
        fmri_dir=base_path + "/participants_data/participants_data_v2021/mini_track/sub{}".format(subject),
        ROI=ROI, activations_dir=base_path + "/i3d_dir/activations", train=False,
        i3d_endpoint=layer_ind_to_activation[str(args.layer_ind)]
    )

    # Env
    input_dim = input_dims[str(args.layer_ind)]
    output_dim = train_dataset.fmri_data.shape[1]
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')


    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dataset = ConcatDataset([train_dataset, test_dataset])

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_train, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size_test, sampler=test_subsampler)

        # Init the neural network
        network = SimpleMLP(input_dim=input_dim, output_dim=output_dim, hidden_sizes=hidden_sizes,
                            activation='relu', device=device)
        # network.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=3e-4, weight_decay=1e-4)

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = []

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, fmri_train = data
                inputs = inputs.to(device)
                fmri_train = fmri_train.to(device)
                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                pred_fmri = network(inputs)

                # Compute loss
                loss = loss_function(pred_fmri, fmri_train)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss.append(loss)

            writer.add_scalar('Loss/train', torch.mean(torch.as_tensor([current_loss])), epoch)

            # Validation
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    # Get inputs
                    inputs, fmri_test = data
                    inputs = inputs.to(device)
                    fmri_test = fmri_test.to(device)

                    # Generate outputs
                    pred_fmri = network(inputs)

                    # Set total and correct
                    score = vectorized_correlation(fmri_test, pred_fmri)

                    # Print fold results
                    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
                    print('--------------------------------')
                    print("Mean correlation for ROI: {} in subject {} is: {}".format(ROI, subject, torch.mean(score)))

            writer.add_scalar('Correlation/val', torch.mean(score), epoch)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = os.path.join(work_dir, f'./model-fold-{fold}.pth')
        torch.save(network.state_dict(), save_path)



def save_activations():
    mode = "rgb"
    work_dir = "./i3d_dir/activations"
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if mode == 'flow':
        fp = "/Users/gautham/src/pytorch-i3d/models/flow_imagenet.pt"
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(fp))
    else:
        # fp = "/Users/gautham/src/pytorch-i3d/models/rgb_imagenet.pt"
        # fp = "/Users/gautham/src/pytorch-i3d/models/rgb_charades.pt"
        fp = "/home/vasan/scratch/pytorch-i3d/models/rgb_charades.pt"
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
        phi = {}
        feat = last_layer_features.view((-1, 7))
        feat = torch.mean(feat, 1)
        phi['Logits'] = feat.cpu().ravel()

        end_points = ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3', 'MaxPool3d_4a_3x3', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c']

        for ep in end_points:
            if ep in ['MaxPool3d_2a_3x3', 'MaxPool3d_3a_3x3']:
                feat = tp_pool(all_phi[ep][0]) # [64, 930]
                feat = torch.mean(feat, 0)
            elif ep == 'MaxPool3d_4a_3x3':
                feat = sp_pool(all_phi[ep][0]) # [480, 105]
                feat = torch.mean(feat, 0)
            else:
                feat = avg_pool(all_phi[ep])
                feat = feat.view((-1, 7))
                feat = torch.mean(feat, 1)
            phi[ep] = feat.cpu().ravel()
            # print(ep, phi[ep].shape)

        fname = os.path.basename(video_fp)[:-4] + ".pkl"
        save_path = os.path.join(work_dir, fname)
        with open(save_path, "wb") as handle:
            pickle.dump(phi, handle)
        print(i_vid, save_path, "{:.3f}".format(time.time() - tic))


if __name__ == '__main__':
    set_one_thread()    # Handle torch shenanigans
    tic = time.time()
    save_activations()
    print("Activation generation took: {:.3f}".format(time.time() - tic))
    #cross_validation_train()
    #print("Training took: {:.3f}".format(time.time() - tic))
