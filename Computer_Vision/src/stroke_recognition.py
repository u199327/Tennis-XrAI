import os

import imutils
import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import ToTensor

from datasets import ThetisDataset, create_train_valid_test_datasets, StrokesDataset
from detection import center_of_box
from utils import get_dtype
import pandas as pd


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.inception_v3(pretrained=True)
        self.feature_extractor.fc = Identity()

    def forward(self, x):
        output = self.feature_extractor(x)
        return output


class LSTM_model(nn.Module):
    """
    Time sequence model for stroke classifying
    """
    def __init__(self, num_classes, input_size=2048, num_layers=3, hidden_size=90, dtype=torch.cuda.FloatTensor):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0, c0 = self.init_state(x.size(0))
        output, (hn, cn) = self.LSTM(x, (h0, c0))
        
        # Take only the last timestep output
        output = output[:, -1, :]  # (batch_size, hidden_size)

        scores = self.fc(output)  # (batch_size, num_classes)
        return scores


    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype))

class GRU_model(nn.Module):
    """
    Time sequence model for stroke classification using GRU.
    """
    def __init__(self, num_classes, input_size=2048, num_layers=3, hidden_size=90, dtype=torch.cuda.FloatTensor):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = self.init_state(x.size(0))
        output, hn = self.GRU(x, h0)
        
        # Take only the last timestep output
        output = output[:, -1, :]  # (batch_size, hidden_size)

        scores = self.fc(output)  # (batch_size, num_classes)
        return scores

    def init_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).type(self.dtype)

class ActionRecognition:
    """
    Stroke recognition model supporting both LSTM and GRU
    """
    def __init__(self, model_saved_state, model_type='LSTM', max_seq_len=55):
        self.dtype = get_dtype()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()
        self.feature_extractor.type(self.dtype)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.max_seq_len = max_seq_len
        self.model_type = model_type

        # Choose model dynamically
        if model_type == 'LSTM':
            self.model = LSTM_model(3, dtype=self.dtype)
        elif model_type == 'GRU':
            self.model = GRU_model(3, dtype=self.dtype)
        else:
            raise ValueError("Invalid model_type. Choose 'LSTM' or 'GRU'.")

        # Load model weights
        saved_state = torch.load(f'saved states/{model_saved_state}', map_location='cpu')
        self.model.load_state_dict(saved_state['model_state'])
        self.model.eval()
        self.model.type(self.dtype)

        self.frames_features_seq = None
        self.box_margin = 150
        self.softmax = nn.Softmax(dim=1)
        self.strokes_label = ['Forehand', 'Backhand', 'Service/Smash']

    def add_frame(self, frame, player_box):
        """
        Extract frame features using feature extractor model and add to sequence.
        """
        box_center = center_of_box(player_box)
        patch = frame[int(box_center[1] - self.box_margin): int(box_center[1] + self.box_margin),
                      int(box_center[0] - self.box_margin): int(box_center[0] + self.box_margin)].copy()
        patch = imutils.resize(patch, 299)
        frame_t = patch.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).type(self.dtype)
        frame_tensor = self.normalize(frame_tensor).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(frame_tensor)
        features = features.unsqueeze(1)

        if self.frames_features_seq is None:
            self.frames_features_seq = features
        else:
            self.frames_features_seq = torch.cat([self.frames_features_seq, features], dim=1)

    def predict_saved_seq(self, clear=True):
        """
        Use saved sequence to predict the stroke.
        """
        with torch.no_grad():
            scores = self.model(self.frames_features_seq)[-1].unsqueeze(0)
            probs = self.softmax(scores).squeeze().cpu().numpy()

        if clear:
            self.frames_features_seq = None
        return probs, self.strokes_label[np.argmax(probs)]

    def predict_stroke(self, frame, player_1_box):
        """
        Predict the stroke for each frame.
        """
        box_center = center_of_box(player_1_box)
        patch = frame[int(box_center[1] - self.box_margin): int(box_center[1] + self.box_margin),
                      int(box_center[0] - self.box_margin): int(box_center[0] + self.box_margin)].copy()
        patch = imutils.resize(patch, 299)
        frame_t = patch.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).type(self.dtype)
        frame_tensor = self.normalize(frame_tensor).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(frame_tensor)
        features = features.unsqueeze(1)

        if self.frames_features_seq is None:
            self.frames_features_seq = features
        else:
            self.frames_features_seq = torch.cat([self.frames_features_seq, features], dim=1)

        if self.frames_features_seq.size(1) > self.max_seq_len:
            self.frames_features_seq = self.frames_features_seq[:, 1:, :]

        with torch.no_grad():
            scores = self.model(self.frames_features_seq)[-1].unsqueeze(0)
            probs = self.softmax(scores).squeeze().cpu().numpy()

        return probs, self.strokes_label[np.argmax(probs)]

def create_features_from_vids():
    """
    Use feature extractor model to create features for each video in the stroke dataset
    """

    dtype = get_dtype()
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()
    feature_extractor.type(dtype)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = StrokesDataset('../dataset/my_dataset/patches/labels.csv', '../dataset/my_dataset/patches/',
                            transform=transforms.Compose([ToTensor(), normalize]), use_features=False)
    batch_size = 32
    count = 0
    for vid in dataset:
        count += 1
        frames = vid['frames']
        print(len(frames))

        features = []
        for batch in frames.split(batch_size):
            batch = batch.type(dtype)
            with torch.no_grad():
                # forward pass
                batch_features = feature_extractor(batch)
                features.append(batch_features.cpu().numpy())

        df = pd.DataFrame(np.concatenate(features, axis=0))

        outfile_path = os.path.join('../dataset/my_dataset/patches/',  os.path.splitext(vid['vid_name'])[0] + '.csv')
        df.to_csv(outfile_path, index=False)

        print(count)

def create_features_from_thetis():
    """
    Use feature extractor model to create features for each video in the THETIS dataset.
    """

    dtype = get_dtype()
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()
    feature_extractor.type(dtype)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = ThetisDataset('./dataset/VIDEO_RGB/THETIS_data.csv', './dataset/VIDEO_RGB',
                            transform=transforms.Compose([ToTensor(), normalize]), use_features=False)
    
    batch_size = 32
    count = 0

    for vid in dataset:
        count += 1
        vid_name = vid['vid_name']
        output_dir = os.path.join('../dataset/features/', vid['vid_folder'])
        os.makedirs(output_dir, exist_ok=True)  # Asegurar que la carpeta existe

        outfile_path = os.path.join(output_dir, os.path.splitext(vid_name)[0] + '.csv')

        if os.path.exists(outfile_path):
            print(f"Skipping {vid_name}, features already exist.")
            continue  # Saltar videos ya procesados

        print(f"Processing video {count}/{len(dataset)} - {vid_name}, {len(vid['frames'])} frames")

        features = []
        for batch in vid['frames'].split(batch_size):
            batch = batch.type(dtype)
            with torch.no_grad():
                batch_features = feature_extractor(batch)
                features.append(batch_features.cpu().numpy())

        # Guardar características en CSV
        df = pd.DataFrame(np.concatenate(features, axis=0))
        df.to_csv(outfile_path, index=False)

        print(f"Saved features to {outfile_path}")
    
if __name__ == "__main__":
    #create_features_from_vids()
    create_features_from_thetis()
    '''batch = None
    video = cv2.VideoCapture('../videos/vid1.mp4')
    while True:
        ret, frame = video.read()
        if ret:
            frame_t = frame.transpose((2, 0, 1)) / 255
            frame_tensor = torch.from_numpy(frame_t).type(dtype)
            frame_tensor = normalize(frame_tensor).unsqueeze(0)
            with torch.no_grad():
                # forward pass
                features = feature_extractor(frame_tensor)
            features = features.unsqueeze(1)
            if batch is None:
                batch = features
            else:
                batch = torch.cat([batch, features], dim=1)
            if batch.size(1) > seq_len:
                # TODO this might be problem, need to get the vector out of gpu
                remove = batch[:,0,:]
                remove.detach().cpu()
                batch = batch[:, 1:, :]
                output = model(batch)
        else:
            break
    video.release()

    cv2.destroyAllWindows()'''
