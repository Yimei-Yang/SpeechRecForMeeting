import torch
from sklearn.linear_model import roc_curve

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class dataset(Dataset):

    def __init__(self, labels_path, features_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.features = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

def initialize(features):quit()
    n = features.size(0)
    p = features.size(1)*features.size(2)
    features = torch.reshape(features, (n, p))
    return [LogisticRegression(), features, labels]

def train(model, x_train, y_train):
    fitted_model = model.fit(x_train, y_train)
    return fitted_model

def evaluate(model, x_test, y_test):
    score = logisticRegr.score(x_test, y_test)
    y_hat = model.predict(y_test)
    [fpr, tpr, thresholds] = roc_curve(labels, y_hat)
    return [score, y_hat, fpr, tpr, thresholds]
