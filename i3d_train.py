import argparse
import torch
import torchvision
import videotransforms
import os
import glob
import pickle
import numpy as np

from pytorch_i3d import InceptionI3d
from perform_encoding import get_fmri
from torch.utils.data import Dataset
from torch import nn
from mlp_models import SimpleMLP
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from utils.ols import vectorized_correlation
from tensorboardX import SummaryWriter

ROIs = ['LOC','FFA','STS','EBA','PPA','V1','V2','V3','V4']


class AlgonautsDataSet(Dataset):
    def __init__(self, activations_dir, fmri_dir, ROI, transform=None, target_transform=None, train=True):
        """

        Args:
            activations_dir:
            fmri_dir: './participants_data/participants_data_v2021'
            ROI:
            transform:
            target_transform:
        """
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
        activations = pickle.load(open(fp, "rb"))
        voxels = self.fmri_data[item]
        voxels = torch.from_numpy(voxels.astype(np.float32))
        return activations, voxels

    def __len__(self):
        return len(self.fmri_data)

def cross_validation_train():
    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('--sub', help='subject number from which real fMRI data will be used', default='01', type=str)
    parser.add_argument('--roi', help='brain region, from which real fMRI data will be used', default='V1', type=str)
    parser.add_argument('--work_dir', help='Store results in this dir', default='./i3d_dir', type=str)
    args = parser.parse_args()

    # Configuration options
    k_folds = 10
    num_epochs = 50
    ROI = args.roi
    subject = args.sub
    loss_function = nn.MSELoss()
    work_dir = args.work_dir + "/sub{}/{}".format(subject, ROI)

    # For fold results
    writer = SummaryWriter(work_dir)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare MNIST dataset by concatenating Train/Test part; we split later.
    # save_activations()
    train_dataset = AlgonautsDataSet(
        fmri_dir="/Users/gautham/src/652_algonauts/participants_data/participants_data_v2021/mini_track/sub{}".format(subject),
        ROI=ROI, activations_dir="/Users/gautham/src/652_algonauts/i3d_dir/activations", train=True
    )

    test_dataset = AlgonautsDataSet(
        fmri_dir="/Users/gautham/src/652_algonauts/participants_data/participants_data_v2021/mini_track/sub{}".format(subject),
        ROI=ROI, activations_dir="/Users/gautham/src/652_algonauts/i3d_dir/activations", train=False
    )

    # Env
    input_dim = 7168
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
            batch_size=32, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100, sampler=test_subsampler)

        # Init the neural network
        network = SimpleMLP(input_dim=input_dim, output_dim=output_dim, hidden_sizes=[2048, 2048],
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
                inputs, targets = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

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

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = os.path.join(work_dir, f'./model-fold-{fold}.pth')
        torch.save(network.state_dict(), save_path)



def save_activations():
    mode = "rgb"
    work_dir = "./i3d_dir/activations/mean"

    if mode == 'flow':
        fp = "/Users/gautham/src/pytorch-i3d/models/flow_imagenet.pt"
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(fp))
    else:
        # fp = "/Users/gautham/src/pytorch-i3d/models/rgb_imagenet.pt"
        fp = "/Users/gautham/src/pytorch-i3d/models/rgb_charades.pt"
        i3d = InceptionI3d(157, in_channels=3, final_endpoint="Logits")
        i3d.load_state_dict(torch.load(fp))

    data_transform = transforms.Compose([videotransforms.CenterCrop(224)])

    all_vids = glob.glob('./participants_data/AlgonautsVideos268_All_30fpsmax/*.mp4')
    all_vids.sort()
    train_vids = all_vids[:1000]
    test_vids = all_vids[1000:]

    for i_vid, video_fp in enumerate(all_vids):
        video_frames, audio_frames, metadata = torchvision.io.read_video(filename=video_fp)
        video_frames = data_transform(video_frames)
        video_frames = torch.permute(video_frames, (3, 0, 1, 2))  # c, t, h, w
        video_frames = torch.unsqueeze(video_frames, dim=0)
        total_frames = video_frames.shape[2]
        indices = np.linspace(0, total_frames - 1, 60, dtype=np.int)
        video_frames = video_frames[:, :, indices, :, :] / 255.

        # pred = i3d(video_frames)
        # phi = torch.mean(pred, axis=-1)
        phi = i3d.extract_features(video_frames).flatten()

        fname = os.path.basename(video_fp)[:-4] + ".pkl"
        save_path = os.path.join(work_dir, fname)
        with open(save_path, "wb") as handle:
            pickle.dump(phi, handle)
        print(i_vid, save_path, phi.shape)


if __name__ == '__main__':
    # save_activations()
    cross_validation_train()
