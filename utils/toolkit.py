import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def accuracy(y_pred, y_true, nb_old, num_classes_per_task=10):
    assert len(y_pred) == len(y_true)
    all_acc = {}
    all_acc['total'] = np.around((y_pred==y_true).sum()*100/len(y_true),2)
    # 按区间累积
    intervals = []
    for cid in range(0, np.max(y_true) + 1, num_classes_per_task):
        idx = np.where((y_true>=cid)&(y_true<cid+num_classes_per_task))[0]
        label = f"{cid:02d}-{cid+num_classes_per_task-1:02d}"
        acc = np.around((y_pred[idx]==y_true[idx]).sum()*100/len(idx),2)
        all_acc[label] = acc
        intervals.append(acc)
    # old/new
    idx_old = np.where(y_true<nb_old)[0]
    idx_new = np.where(y_true>=nb_old)[0]
    all_acc['old'] = 0 if len(idx_old)==0 else np.around((y_pred[idx_old]==y_true[idx_old]).sum()*100/len(idx_old),2)
    all_acc['new'] = np.around((y_pred[idx_new]==y_true[idx_new]).sum()*100/len(idx_new),2)
    all_acc['class_acc'] = intervals
    return all_acc

def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
