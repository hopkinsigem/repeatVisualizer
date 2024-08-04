import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def preprocess_and_merge_data(probabilities_folder, annotations_folder, output_folder, length_75th_percentile):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prob_files = sorted(glob.glob(os.path.join(probabilities_folder, '*.npy')))
    annot_files = sorted(glob.glob(os.path.join(annotations_folder, '*.npy')))

    merged_prob_data = []
    merged_annot_data = []

    for prob_file, annot_file in zip(prob_files, annot_files):
        prob_data = np.load(prob_file)
        annot_data = np.load(annot_file)

        resized_prob_data = np.zeros((10, length_75th_percentile))
        resized_annot_data = np.zeros((10, length_75th_percentile))

        for i in range(prob_data.shape[0]):
            resized_prob_data[i, :] = np.interp(
                np.linspace(0, prob_data.shape[1], length_75th_percentile),
                np.arange(prob_data.shape[1]),
                prob_data[i, :]
            )
            resized_annot_data[i, :] = np.interp(
                np.linspace(0, annot_data.shape[1], length_75th_percentile),
                np.arange(annot_data.shape[1]),
                annot_data[i, :]
            )

        merged_prob_data.append(resized_prob_data)
        merged_annot_data.append(resized_annot_data)

    merged_prob_data = np.stack(merged_prob_data, axis=0)
    merged_annot_data = np.stack(merged_annot_data, axis=0)

    np.save(os.path.join(output_folder, 'merged_probabilities.npy'), merged_prob_data)
    np.save(os.path.join(output_folder, 'merged_annotations.npy'), merged_annot_data)

probabilities_folder = r'C:\Users\User\Downloads\igem\dataset\probabilities_dataset'
annotations_folder = r'C:\Users\User\Downloads\igem\dataset\annotations_dataset'
output_folder = 'merged_data'

sequence_lengths = []
prob_files = glob.glob(os.path.join(probabilities_folder, '*.npy'))

for file in prob_files:
    data = np.load(file)
    sequence_lengths.append(data.shape[1])

length_75th_percentile = int(np.percentile(sequence_lengths, 75))
print(f"75th percentile length: {length_75th_percentile}")

preprocess_and_merge_data(probabilities_folder, annotations_folder, output_folder, length_75th_percentile)

class AminoAcidAnnotator(nn.Module):
    def __init__(self, input_size, output_size):
        super(AminoAcidAnnotator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * length_75th_percentile, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = 10  
output_size = length_75th_percentile  
model = AminoAcidAnnotator(input_size, output_size)

def train_neural_network(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()


        average_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(average_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        average_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss}, Val Loss: {average_val_loss}')

    return train_losses, val_losses

merged_prob_file = 'merged_data/merged_probabilities.npy'
merged_annot_file = 'merged_data/merged_annotations.npy'

merged_prob_data = np.load(merged_prob_file)
merged_annot_data = np.load(merged_annot_file)

prob_train, prob_val, annot_train, annot_val = train_test_split(merged_prob_data, merged_annot_data, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(prob_train, dtype=torch.float32), torch.tensor(annot_train, dtype=torch.float32))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(prob_val, dtype=torch.float32), torch.tensor(annot_val, dtype=torch.float32))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = AminoAcidAnnotator(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses, val_losses = train_neural_network(model, train_loader, val_loader, criterion, optimizer, num_epochs)

plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
