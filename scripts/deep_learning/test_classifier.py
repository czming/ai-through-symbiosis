# import torch
# import torch.nn as nn
# import time
# from torchvision import transforms
# import os
# from torchsummary import summary
# from torch.optim import SGD, Adam
# from EgoObjectDataset import EgoObjectClassificationDataset
# from models.resnet import Resnet18Classifier, Resnet34Classifier, Resnet9Classifier
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from tqdm import tqdm
# from torch.nn.modules.distance import PairwiseDistance
# import numpy as np
# from sklearn import metrics
# import matplotlib.pyplot as plt
# from collections import defaultdict, Counter
# import pandas as pd
# # torch.backends.cudnn.enabled = False
#
#
# # from losses import TripletLoss, Accuracy
#
# def get_model(pretrained=False, num_classes=10):
#     model = Resnet9Classifier(
#         num_classes=num_classes,
#         pretrained=pretrained
#     )
#     return model
#
#
# def forward_pass(imgs, model, batch_size):
#     imgs = imgs.cuda()
#     embeddings = model(imgs)
#
#     split_size = int(embeddings.shape[0] / 3)
#
#     # Split the embeddings into Anchor, Positive, and Negative embeddings
#     anc_embeddings = embeddings[:split_size]
#     pos_embeddings = embeddings[split_size: split_size * 2]
#     neg_embeddings = embeddings[split_size * 2:]
#
#     return anc_embeddings, pos_embeddings, neg_embeddings, model
#
#
# def train():
#     image_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         # transforms.RandomApply([transforms.RandomResizedCrop((256,256))], p = 0.2),
#         transforms.Resize((64, 64)),
#         transforms.RandomRotation(45),
#         transforms.RandomHorizontalFlip(0.3),
#         transforms.RandomVerticalFlip(0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.6071, 0.4609, 0.3944],
#             std=[0.2457, 0.2175, 0.2129]
#         )
#     ])
#     learning_rate = 0.0001
#     batch_size = 64
#     dataset = EgoObjectClassificationDataset('data/labeled_objects_new.csv', data_, transform=image_transforms)
#     train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     model_architecture = 'resnet34'
#     model = get_model(pretrained=True)
#     print(model)
#     model = model.cuda()
#     model.train()
#     # margin = 0.2
#     # l2_distance = PairwiseDistance(p=2)
#     # progress_bar = enumerate(tqdm(train_dataloader))
#     optimizer_model = optim.Adam(
#         params=model.parameters(),
#         lr=learning_rate,
#         # momentum=0.9,
#         # dampening=0,
#         # nesterov=False,
#         # weight_decay=1e-5
#     )
#
#     lr_scheduler = StepLR(optimizer_model, step_size=20, gamma=0.5)
#
#     criterion = torch.nn.CrossEntropyLoss()
#     # criterion = torch.nn.NLLLoss()
#     total_epochs = 30
#     cur_epoch = 0
#     # print(len(train_dataloader))
#     # exit()
#     while cur_epoch < total_epochs:
#
#         print (f"lr: {lr_scheduler.get_last_lr()}")
#
#         time_now = time.time()
#         total_loss = 0
#         num_valid_training_triplets = 0
#         total = 0
#         correct = 0
#         cur_epoch += 1
#         epoch_loss = 0
#         for batch_idx, (batch_sample) in enumerate(train_dataloader):
#             # Forward pass - compute embeddings
#             imgs = batch_sample[0].cuda()
#             # print(batch_sample[1])
#             labels = torch.tensor(batch_sample[1]).cuda()
#             preds = model(imgs)
#             # print (preds, labels)
#             _, predicted = torch.max(preds.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             # print(anc_embeddings.shape,pos_embeddings.shape, neg_embeddings.shape )
#             loss = criterion(preds, labels)
#             optimizer_model.zero_grad()
#             loss.backward()
#             optimizer_model.step()
#             epoch_loss += loss.item()
#             if batch_idx % (len(train_dataloader) // 10) == 0:
#                 # only want to see like 10 batches per epoch
#                 print('Epoch {} Batch {}/{}:\tLoss: {}'.format(
#                     cur_epoch,
#                     batch_idx,
#                     len(train_dataloader),
#                     loss.item()
#                 )
#                 )
#         total_acc = correct / total
#         print('###############################')
#         print('Epoch {}:\tEpoch Loss: {}\tEpoch Accuracy: {}'.format(
#             cur_epoch,
#             epoch_loss,
#             total_acc
#         )
#         )
#         print('###############################')
#
#         # step for lr_scheduler after one epoch
#         lr_scheduler.step()
#
#         state = {
#             'epoch': cur_epoch,
#             'batch_size_training': batch_size,
#             'model_state_dict': model.state_dict(),
#             'model_architecture': model_architecture,
#             'optimizer_model_state_dict': optimizer_model.state_dict(),
#         }
#         if cur_epoch % 5 == 0:
#             torch.save(state, 'model_training_checkpoints/model_{}_classifier_epoch_{}.pt'.format(
#                 model_architecture,
#                 cur_epoch
#             )
#             )
#
#
# def test():
#     image_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         # transforms.RandomApply([transforms.RandomResizedCrop((256,256))], p = 0.2),
#         transforms.Resize((64, 64)),
#         # transforms.RandomRotation(45),
#         # transforms.RandomHorizontalFlip(0.3),
#         # transforms.RandomVerticalFlip(0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.6071, 0.4609, 0.3944],
#             std=[0.2457, 0.2175, 0.2129]
#         )
#     ])
#     accs = []
#     for epochs in [30]:
#         learning_rate = 0.075
#         model_path = f'../model_training_checkpoints/model_resnet9_classifier_epoch_30_lr_0.001.pt'
#         #'../model_training_checkpoints/model_{}_classifier_epoch_{}_lr_{}.pt'
#         batch_size = 512
#         dataset = EgoObjectClassificationDataset('../data/labeled_objects_test.csv', test=True, transform=image_transforms)
#         train_dataloader = DataLoader(dataset, batch_size=batch_size)
#         model_architecture = 'resnet9'
#         model = get_model()
#         model.load_state_dict(torch.load(model_path)['model_state_dict'])
#         print(model)
#         model = model.cuda()
#         model.eval()
#         # margin = 0.2
#         # l2_distance = PairwiseDistance(p=2)
#         # progress_bar = enumerate(tqdm(train_dataloader))
#         # optimizer_model = optim.SGD(
#         #     params=model.parameters(),
#         #     lr=learning_rate,
#         #     momentum=0.9,
#         #     dampening=0,
#         #     nesterov=False,
#         #     weight_decay=1e-5
#         # )
#         # criterion = torch.nn.CrossEntropyLoss()
#         # criterion = torch.nn.NLLLoss()
#         # total_epochs = 1000
#         # cur_epoch = 0
#         # print(len(train_dataloader))
#         # exit()
#         # while cur_epoch < total_epochs:
#         time_now = time.time()
#         total_loss = 0
#         num_valid_training_triplets = 0
#         total = 0
#         correct = 0
#         gt_list = []
#         preds_list = []
#         for batch_idx, (batch_sample) in enumerate(train_dataloader):
#             # Forward pass - compute embeddings
#             imgs = batch_sample[0].cuda()
#             # print(batch_sample[1])
#             labels = torch.tensor(batch_sample[1]).cuda()
#             preds = model(imgs)
#             _, predicted = torch.max(preds.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             preds_list.extend(list(predicted.detach().cpu().numpy()))
#             gt_list.extend(batch_sample[1])
#         total_acc = correct / total
#         print('###############################')
#         print('Test Accuracy: {}'.format(
#             total_acc
#         )
#         )
#         print('###############################')
#         accs.append((epochs, total_acc))
#         confusion_matrix = metrics.confusion_matrix(gt_list, preds_list)
#         conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])
#         conf_mat_norm = np.around(conf_mat_norm, decimals=2)
#         # unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
#         cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm,
#                                                     display_labels=['red', 'green', 'blue', 'darkblue', \
#                                                                     'darkgreen', 'orange', 'alligatorclip', 'yellow', \
#                                                                     'clear', 'candle'])
#
#         cm_display.plot(cmap=plt.cm.Blues)
#
#         plt.xticks(rotation=90)
#
#         plt.tight_layout()
#
#         # plt.show()
#
#         plt.savefig(f'results_resnet9_test_classifier_epochs_{epochs}.png')
#
#         print(accs)
#
# def test_per_pick():
#
#     train_picklists = set(pd.read_csv('../data/labeled_objects_new.csv')['picklist'])
#     train_picklists = set([int(x[-3:]) for x in train_picklists])
#     print(train_picklists)
#     full_set = set(list(range(136, 235))) - set(train_picklists)
#     print(full_set)
#
#
#     image_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         # transforms.RandomApply([transforms.RandomResizedCrop((256,256))], p = 0.2),
#         transforms.Resize((64, 64)),
#         # transforms.RandomRotation(45),
#         # transforms.RandomHorizontalFlip(0.3),
#         # transforms.RandomVerticalFlip(0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.6071, 0.4609, 0.3944],
#             std=[0.2457, 0.2175, 0.2129]
#         )
#     ])
#     accs = []
#     model_path = f'../model_training_checkpoints/model_resnet9_classifier_epoch_30_lr_0.001.pt'
#     #'../model_training_checkpoints/model_{}_classifier_epoch_{}_lr_{}.pt'
#     batch_size = 512
#     dataset = EgoObjectClassificationDataset('../data/labeled_objects_new_gt.csv', test=False, numbers=True, picklists=list(full_set), transform=image_transforms)
#     train_dataloader = DataLoader(dataset, batch_size=batch_size)
#     model_architecture = 'resnet9'
#     model = get_model()
#     model.load_state_dict(torch.load(model_path)['model_state_dict'])
#     print(model)
#     model = model.cuda()
#     model.eval()
#     time_now = time.time()
#     total_loss = 0
#     total = 0
#     correct = 0
#     gt_list = []
#     preds_list = []
#     picklist_map = defaultdict(list)
#     true_map = defaultdict(list)
#     for batch_idx, (batch_sample) in enumerate(train_dataloader):
#         # Forward pass - compute embeddings
#         imgs = batch_sample[0][0].cuda()
#         frames = list(batch_sample[0][1])
#         picklists = list(batch_sample[0][2])
#         # print(batch_sample[1])
#         labels = torch.tensor(batch_sample[1]).cuda()
#         preds = model(imgs)
#         preds = list(torch.argmax(preds, dim=-1))
#         labels = list(labels)
#         for i in range(len(preds)):
#             picklist_map[picklists[i]].append((frames[i].item(), preds[i].cpu().item()))
#             true_map[picklists[i]].append((frames[i].item(), labels[i].cpu().item()))
#     for k, v in picklist_map.items():
#         picklist_map[k] = sorted(v)
#     for k, v in true_map.items():
#         picklist_map[k] = sorted(v)
#
#     for k, v in picklist_map.items():
#         new_v = []
#         curr = [v[0][1]]
#         start = v[0][0]
#         for i in range(1, len(v)):
#             if v[i][0] - v[i-1][0] == 1:
#                 curr.append(v[i][1])
#             else:
#                 max_occ = Counter(curr).most_common(1)[0][0]
#                 new_v.append((start, max_occ))
#                 curr = [v[i][1]]
#                 start = v[i][0]
#         max_occ = Counter(curr).most_common(1)[0][0]
#         new_v.append((start, max_occ))
#         picklist_map[k] = new_v
#
#     for k, v in true_map.items():
#         new_v = []
#         curr = [v[0][1]]
#         start = v[0][0]
#         for i in range(1, len(v)):
#             if v[i][0] - v[i-1][0] == 1:
#                 curr.append(v[i][1])
#             else:
#                 max_occ = Counter(curr).most_common(1)[0][0]
#                 new_v.append((start, max_occ))
#                 curr = [v[i][1]]
#                 start = v[i][0]
#         max_occ = Counter(curr).most_common(1)[0][0]
#         new_v.append((start, max_occ))
#         true_map[k] = new_v
#
#     final_pl_map = {}
#     for k, v in picklist_map.items():
#         for frame_no, pred in v:
#             final_pl_map[(k, frame_no)] = pred
#     final_true_map = {}
#     for k, v in true_map.items():
#         for frame_no, label in v:
#             final_true_map[(k, frame_no)] = label
#
#     true = []
#     preds = []
#     for k in final_pl_map:
#         preds.append(final_pl_map[k])
#         true.append(final_true_map[k])
#
#     print(preds)
#     print(true)
#     total = 0
#     correct = 0
#     for p, t in zip(preds, true):
#         correct += p==t
#         total += 1
#     # print(true_map)
#     total_acc = correct / total
#     print('###############################')
#     print('Test Accuracy: {}'.format(
#         total_acc
#     )
#     )
#     print('###############################')
#     # accs.append((epochs, total_acc))
#     confusion_matrix = metrics.confusion_matrix(true, preds)
#     conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])
#     conf_mat_norm = np.around(conf_mat_norm, decimals=2)
#     # unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm,
#                                                 display_labels=['red', 'green', 'blue', 'darkblue', \
#                                                                 'darkgreen', 'orange', 'alligatorclip', 'yellow', \
#                                                                 'clear', 'candle'])
#
#     cm_display.plot(cmap=plt.cm.Blues)
#
#     plt.xticks(rotation=90)
#
#     plt.tight_layout()
#
#     # plt.show()
#
#     plt.savefig(f'results_resnet9_test_classifier_grouped.png')
#
#     # print(accs)
#
# if __name__ == '__main__':
#     # train()
#     # test()
#     test_per_pick()