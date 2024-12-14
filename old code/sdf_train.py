import torch 
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader, Subset
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from AutoEncoder.encoder import PointCloudAutoEncoder
from Helpers.SDFDataset import SDFDataset
from Helpers.PointCloudDataset import PointCloudDataset
from sdf_models import SDFRegressionModel_3L, SDFRegressionModel_5L, SDFRegressionModel_7L, SDFRegressionModel_9L
from sdf_models import SDFClassificationModel_3L, SDFClassificationModel_5L, SDFClassificationModel_7L, SDFClassificationModel_9L

# Device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f'Using: {device}')

# Example of a batch of point clouds being encoded to latent reps
point_size = 3072
latent_shape = 512
encoder = PointCloudAutoEncoder(model_type= '800T', point_size= str(point_size), path_to_weight_dir= '../AutoEncoder/')
encoder.eval() # Let torch know that it doesn't need to store activations as there will be no backward pass

# Combined PC and SDF Dataset and DataLoader
# object_classes=None
object_classes=['glass_box']
# object_classes=['airplane','bathtub','bed','bench','bookshelf','bottle','car']
# sdf_base_dir='sampled_vdbs/sampled_vdbs'
sdf_base_dir='../../sampled_points_easier/vdbs'
point_cloud_base_dir= "../../Data/ModelNet40"
train_dataset = SDFDataset(sdf_base_dir, point_cloud_base_dir, point_size, 'train', object_classes)
test_dataset = SDFDataset(sdf_base_dir, point_cloud_base_dir, point_size, 'test', object_classes)

test_size = len(test_dataset)
split_idx = test_size // 2
indices = list(range(test_size))
# val_dataset_3072 = Subset(test_dataset, indices[:split_idx])
# test_dataset_3072 = Subset(test_dataset, indices[split_idx:])

# Save Dataset and Load DataLoader
test_dataset_3072 = torch.load(f'pc_dataset_{point_size}_points_all_objects_test_dataset_3072_order_points_True',  weights_only=False)
val_dataset_3072 = torch.load(f'pc_dataset_{point_size}_points_all_objects_val_dataset_3072_order_points_True',  weights_only=False)

# torch.save(test_dataset_3072, f'pc_dataset_{point_size}_points_all_objects_test_dataset_3072_order_points_True')
# torch.save(val_dataset_3072, f'pc_dataset_{point_size}_points_all_objects_val_dataset_3072_order_points_True')

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset_3072, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset_3072, batch_size=64, shuffle=True)



############################################
#### CLASSIFICATION TRAINING FUNCTION   ####
############################################
# Function to calculate accuracy 
def calculate_accuracy(outputs, targets): 
    preds = torch.round(torch.sigmoid(outputs))
    correct = (preds == targets).float() 
    accuracy = torch.sum(correct) / (correct.shape[0]*correct.shape[1])
    return accuracy

def get_latent_rep_from_pc(point_cloud):
    point_cloud = point_cloud.permute(0,2,1)
    latent_rep = encoder(point_cloud) # (batch_size=64, latent_dim=512)
    latent_rep = latent_rep.unsqueeze(1).repeat(1, 10000, 1) # (64, 10000, 512)
    return latent_rep

def train_model(model, criterion, num_epochs, train_loader, val_loader, learning_rate=0.0001):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, val_loss, total_train_acc, total_val_acc = 0, 0, 0.0, 0.0
        for data in train_loader:
            # This loop ensures that the point cloud filename corresponds to the sdf filename, with dir and ext info removed
            # for path_to_pc, path_to_sdf in zip(data['pc_filename'], data['sdf_filename']):
            #     pc_filename = path_to_pc.split('/')[-1].split('.')[0]
            #     sdf_filename = path_to_sdf.split('/')[-1].split('.')[0]
            #     if pc_filename!= sdf_filename:
            #         print(f"PointCloudDataset {pc_filename} and SDFDataset {sdf_filename} filenames need to correspond")
            #         break
            # print(f'{data["sdf_filename"]}   {data["pc_filename"]}')
            sample_sdf, point_cloud, sdf_labels = data['sdf_points'], data['point_clouds'], data['sdf_labels']
            if point_cloud.shape[0] == 64:
                latent_rep = get_latent_rep_from_pc(point_cloud).to(device)
                sdf_point = sample_sdf.to(device)
                # Convert to -1 if negative and 1 if positive - Shape (64,10000)
                labels = torch.where(sdf_labels < 0, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)).to(device)
                # print(f'labels: {labels}')
                optimizer.zero_grad()
                outputs = model(sdf_point, latent_rep).squeeze(-1) # shape (64,10000,1)
                loss = criterion(outputs, labels)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                total_train_acc += calculate_accuracy(outputs, labels.float()).item()
        model.eval()
        for data in val_loader:
            sample_sdf, point_cloud, sdf_labels = data['sdf_points'], data['point_clouds'], data['sdf_labels']
            if point_cloud.shape[0] == 64:
                with torch.no_grad():
                    latent_rep = get_latent_rep_from_pc(point_cloud).to(device)
                    sdf_point=sample_sdf.to(device)
                    labels = torch.where(sdf_labels < 0, torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)).to(device)
                    outputs = model(sdf_point, latent_rep).squeeze(-1)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    total_val_acc += calculate_accuracy(outputs, labels.float()).item()
        model.train()
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(val_loader)) 
        train_accs.append(total_train_acc/len(train_loader))
        val_accs.append(total_val_acc/len(val_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}, Train Acc: {total_train_acc/len(train_loader)}, Val Acc: {total_val_acc/len(val_loader)}')
    print("Training complete!")
    return train_losses, val_losses, train_accs, val_accs

#####################################################################
# Call SDFClassificationModel Training Function With Varying Models #
#####################################################################
import json

models = {
    "3L": SDFClassificationModel_3L,
    "5L": SDFClassificationModel_5L,
    "7L": SDFClassificationModel_7L,
    "9L": SDFClassificationModel_9L
}

# Hyperparameters
criterion = nn.BCEWithLogitsLoss()
input_dim = 3
latent_dim = 512
hidden_dim = 256
learning_rate = 0.0001
num_epochs = 3

training_results_smallest_lr = {}

for model_name, model_class in models.items():
    model = model_class(input_dim, latent_dim, hidden_dim).to(device)
    train_losses, val_losses, train_accs, val_accs = train_model(model, criterion, num_epochs, train_dataloader, val_dataloader, learning_rate) 
    training_results_smallest_lr[model_name] = {'train_losses': train_losses, 
                                    'val_losses': val_losses, 
                                    'train_accs': train_accs, 
                                    'val_accs': val_accs}
with open('./results/sdf_classifier_512_256_lr_0.0001.json', 'w') as f:
    json.dump(training_results_smallest_lr, f)

learning_rate = 0.0005
training_results_medium_lr = {}
for model_name, model_class in models.items():
    model = model_class(input_dim, latent_dim, hidden_dim).to(device)
    train_losses, val_losses, train_accs, val_accs = train_model(model, criterion, num_epochs, train_dataloader, val_dataloader, learning_rate) 
    training_results_medium_lr[model_name] = {'train_losses': train_losses, 
                                    'val_losses': val_losses, 
                                    'train_accs': train_accs, 
                                    'val_accs': val_accs}
with open('./results/sdf_classifier_512_256_lr_0.0001.json', 'w') as f:
    json.dump(training_results_medium_lr, f)

learning_rate = 0.001
training_results_largest_lr= {}
for model_name, model_class in models.items():
    model = model_class(input_dim, latent_dim, hidden_dim).to(device)
    train_losses, val_losses, train_accs, val_accs = train_model(model, criterion, num_epochs, train_dataloader, val_dataloader, learning_rate) 
    training_results_largest_lr[model_name] = {'train_losses': train_losses, 
                                    'val_losses': val_losses, 
                                    'train_accs': train_accs, 
                                    'val_accs': val_accs}
with open('./results/sdf_classifier_512_256_lr_0.0001.json', 'w') as f:
    json.dump(training_results_largest_lr, f)

############################################################
# Plot Accuracies And Losses For Each Model Training Above #
############################################################
import matplotlib.pyplot as plt
epochs = list(range(num_epochs))

for model_name, results in training_results_smallest_lr.items():
    plt.plot(epochs, results['train_losses'], label=model_name+'_train')
    plt.plot(epochs, results['val_losses'], label=model_name+'_val')
plt.title('Training and Val Losses With 0.0001 LR')
plt.legend()
plt.show()

for model_name, results in training_results_medium_lr.items():
    plt.plot(epochs, results['train_losses'], label=model_name+'_train')
    plt.plot(epochs, results['val_losses'], label=model_name+'_val')
plt.title('Training and Val Losses With 0.0005 LR')
plt.legend()
plt.show()

for model_name, results in training_results_largest_lr.items():
    plt.plot(epochs, results['train_losses'], label=model_name+'_train')
    plt.plot(epochs, results['val_losses'], label=model_name+'_val')
plt.title('Training and Val Losses With 0.001 LR')
plt.legend()
plt.show()

for model_name, results in training_results_smallest_lr.items():
    plt.plot(epochs, results['train_accs'], label=model_name+'_train')
    plt.plot(epochs, results['val_accs'], label=model_name+'_val')
plt.title('Training and Val Accuracies With 0.0001 LR')
plt.legend()
plt.show()

for model_name, results in training_results_medium_lr.items():
    plt.plot(epochs, results['train_accs'], label=model_name+'_train')
    plt.plot(epochs, results['val_accs'], label=model_name+'_val')
plt.title('Training and Val Accuracies With 0.0005 LR')
plt.legend()
plt.show()

for model_name, results in training_results_largest_lr.items():
    plt.plot(epochs, results['train_accs'], label=model_name+'_train')
    plt.plot(epochs, results['val_accs'], label=model_name+'_val')
plt.title('Training and Val Accuracies With 0.001 LR')
plt.legend()
plt.show()


######################################
# Regression Models, Loss, Optimizer #
######################################
# model_3L = SDFRegressionModel_3L(input_dim, latent_dim, hidden_dim).to(device)
# model_5L = SDFRegressionModel_5L(input_dim, latent_dim, hidden_dim).to(device)
# model_7L = SDFRegressionModel_7L(input_dim, latent_dim, hidden_dim).to(device)
# model_9L = SDFRegressionModel_9L(input_dim, latent_dim, hidden_dim).to(device)

# optimizer_3L = optim.AdamW(model_3L.parameters(), lr=learning_rate)
# optimizer_5L = optim.AdamW(model_5L.parameters(), lr=learning_rate)
# optimizer_7L = optim.AdamW(model_7L.parameters(), lr=learning_rate)
# optimizer_9L = optim.AdamW(model_9L.parameters(), lr=learning_rate)
# criterion = nn.MSELoss()


#########################################
#    Regression Training Function       #
#########################################
# def train_model(model, optimizer, num_epochs, train_loader, test_loader=None):
#     train_losses = []
#     for epoch in range(num_epochs):
#         train_loss = 0
#         for data in train_loader:
#             # This loop ensures that the point cloud filename corresponds to the sdf filename, with dir and ext info removed
#             # for path_to_pc, path_to_sdf in zip(data['pc_filename'], data['sdf_filename']):
#             #     pc_filename = path_to_pc.split('/')[-1].split('.')[0]
#             #     sdf_filename = path_to_sdf.split('/')[-1].split('.')[0]
#             #     if pc_filename!= sdf_filename:
#             #         print(f"PointCloudDataset {pc_filename} and SDFDataset {sdf_filename} filenames need to correspond")
#             #         break
#             print(f'{data["sdf_filename"]}   {data["pc_filename"]}')
#             sample_sdf, point_cloud, sdf_labels = data['sdf_points'], data['point_clouds'], data['sdf_labels']
#             if point_cloud.shape[0] == 64:
#                 point_cloud = point_cloud.permute(0,2,1)
#                 latent_rep = encoder(point_cloud) # (64, 512)
#                 latent_rep = latent_rep.unsqueeze(1).repeat(1, 10000, 1).to(device) # (64, 10000, 512)
#                 sdf_point=sample_sdf.to(device)
#                 labels = sdf_labels.to(device) # Shape (64,10000)

#                 optimizer.zero_grad()
#                 outputs = model(sdf_point, latent_rep) # shape (64,10000,1)
#                 outputs = outputs.squeeze(-1)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 train_loss += loss.item()
#                 optimizer.step()
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}')
#         train_losses.append(train_loss/len(train_loader))
#     print("Training complete!")
#     return train_losses


#################################################################
# Call SDFRegressionModel Training Function With Varying Models #
#################################################################
# train_losses = train_model(model_3L, optimizer_3L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_3L_lr_0.0001.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_5L, optimizer_5L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_5L_lr_0.0001.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_7L, optimizer_7L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_7L_lr_0.0001.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_9L, optimizer_9L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_9L_lr_0.0001.txt', 'w') as f:
#     f.writelines(str(train_losses))

# learning_rate = 0.0005
# optimizer_3L = optim.AdamW(model_3L.parameters(), lr=learning_rate)
# optimizer_5L = optim.AdamW(model_5L.parameters(), lr=learning_rate)
# optimizer_7L = optim.AdamW(model_7L.parameters(), lr=learning_rate)
# optimizer_9L = optim.AdamW(model_9L.parameters(), lr=learning_rate)

# train_losses = train_model(model_3L, optimizer_3L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_3L_lr_0.0005.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_5L, optimizer_5L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_5L_lr_0.0005.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_7L, optimizer_7L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_7L_lr_0.0005.txt', 'w') as f:
#     f.writelines(str(train_losses))
# train_losses = train_model(model_9L, optimizer_9L, num_epochs, train_dataloader)
# print(str(train_losses)+'\n')
# with open('./results/easier_model_all_objects_9L_lr_0.0005.txt', 'w') as f:
#     f.writelines(str(train_losses))
