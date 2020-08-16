from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from sklearn.externals import joblib


import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(".")
from reid import datasets
from reid import models
from reid.trainers import ClusterBaseTrainer, DualRefineTrainer
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor, TrainPreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.rerank import compute_jaccard_dist
from reid.loss.spreadloss import SpreadLoss


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(TrainPreprocessor(train_set, root=dataset.images_dir,
                                             transform=train_transformer, mutual=False),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args, classes):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=classes)

    model.cuda()
    model = nn.DataParallel(model)

    initial_weights = load_checkpoint(args.init)
    copy_state_dict(initial_weights['state_dict'], model)

    return model


def reassign_labels(all_feats, cluster_feats, R):

    all_feats = F.normalize(all_feats, dim=1)
    protos = []
    star_time = time.time()
    for id in sorted(cluster_feats.keys()):
        c_feats = torch.stack(cluster_feats[id])
        c_feats = F.normalize(c_feats, dim=1).numpy()
        if c_feats.shape[0]>R:
            kmeans = KMeans(n_clusters=R, random_state=0).fit(c_feats)
            c_centers = kmeans.cluster_centers_
        else:
            mean_c_feats = np.mean(c_feats, axis=0, keepdims=True)
            c_centers = np.repeat(mean_c_feats, R, axis=0)
        protos.append(c_centers)
    protos = torch.from_numpy(np.array(protos, dtype=np.float32))
    protos = F.normalize(protos, dim=-1)

    scores = torch.matmul(protos, all_feats.t())
    scores = torch.sum(scores, dim=1)
    _, sorted_index = torch.sort(scores.t(), dim=1, descending=True)
    end_time = time.time()
    # print('Kmeans cost time: ', end_time-star_time)

    clean_labels = sorted_index[:, 0]

    return clean_labels


def calScores(clusters, labels):
    """
    compute pair-wise precision pair-wise recall
    """
    from scipy.special import comb
    if len(clusters) == 0:
        return 0, 0
    else:
        curCluster = []
        for curClus in clusters.values():
            curCluster.append(labels[curClus])
        TPandFP = sum([comb(len(val), 2) for val in curCluster])
        TP = 0
        for clusterVal in curCluster:
            for setMember in set(clusterVal):
                if sum(clusterVal == setMember) < 2: continue
                TP += comb(sum(clusterVal == setMember), 2)
        FP = TPandFP - TP
        # FN and TN
        TPandFN = sum([comb(labels.tolist().count(val), 2) for val in set(labels)])
        FN = TPandFN - TP
        # cal precision and recall
        precision, recall = TP / (TP + FP), TP / (TP + FN)
        fScore = 2 * precision * recall / (precision + recall)
        return precision, recall, fScore


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source = get_data(args.dataset_source, args.data_dir)
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers, testset=dataset_source.train)

    # Create model
    model = create_model(args, len(dataset_target.train))

    # Evaluator
    evaluator = Evaluator(model)

    ### save F-scores
    # Fscores = collections.defaultdict(list)

    for epoch in range(args.epochs):
        dict_f, _ = extract_features(model, tar_cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f.values()))
        # create memory bank model and initialize it
        if epoch==0:
            num_tgt = len(dataset_target.train)
            target_features = cf.numpy()
            init_memory = target_features / np.linalg.norm(target_features, axis=1, keepdims=True)

            model_spread = SpreadLoss(2048, num_tgt, init_memory, m=args.m, knn=args.knn)
            model_spread = model_spread.cuda()

        if (args.lambda_value>0):
            dict_f, _ = extract_features(model, sour_cluster_loader, print_freq=50)
            cf_s = torch.stack(list(dict_f.values()))
            rerank_dist = compute_jaccard_dist(cf, lambda_value=args.lambda_value, source_features=cf_s, use_gpu=args.rr_gpu).numpy()
        else:
            rerank_dist = compute_jaccard_dist(cf, use_gpu=args.rr_gpu).numpy()

        if (epoch==0):
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            rho = args.rho
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        args.num_clusters = num_ids
        print('\n Clustered into {} classes \n'.format(args.num_clusters))

        # generate new dataset and calculate cluster centers
        new_dataset = []
        noise_labels = []
        cluster_centers = collections.defaultdict(list)
        cluster_feats = collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(dataset_target.train, labels)):
            new_dataset.append((fname,label,cid))
            noise_labels.append(label)
            if label != -1:
                cluster_centers[label].append(cf[i])
                cluster_feats[label].append(cf[i])

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        model.module.classifier.weight.data[:args.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters, trainset=new_dataset)

        # learning rate decay
        step_size = args.epochs_decay
        new_lr = args.lr * (0.1 ** (epoch // step_size))

        # Optimizer
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": new_lr, "weight_decay": args.weight_decay}]

        # optimizer for instant memory bank
        for key, value in model_spread.named_parameters():
            # print(key,value)
            params += [{"params": [value], "lr": new_lr, "weight_decay": args.weight_decay}]

        optimizer = torch.optim.Adam(params)

        clean_labels = reassign_labels(cf, cluster_feats, args.R)

        ### calculate F1-score of the clusters
        # realIDs, coarseIDs, fineIDs = collections.defaultdict(list), [], []
        # counter = 0
        # print(labels.shape, clean_labels.shape)
        # for i, ((fname, real_label, cid), coase_label, fine_label) in enumerate(
        #         zip(dataset_target.train, labels, clean_labels.numpy())):
        #
        #     if coase_label != -1:
        #         realIDs[real_label].append(counter)
        #         coarseIDs.append(coase_label)
        #         fineIDs.append(fine_label)
        #         counter += 1

        # c_precision, c_recall, c_fscore = calScores(realIDs, np.asarray(coarseIDs))
        # f_precision, f_recall, f_fscore = calScores(realIDs, np.asarray(fineIDs))

        # print('Coarse-clustering: precision={}, recall={}, fscore={}'.format(c_precision, c_recall, c_fscore))
        # print('Fine-clustering: precision={}, recall={}, fscore={}'.format(f_precision, f_recall, f_fscore))

        # Fscores['coarse_F1'].append(c_fscore)
        # Fscores['fine_F1'].append(f_fscore)
        #
        # Fscores['coarse_P'].append(c_precision)
        # Fscores['coarse_R'].append(c_recall)
        # Fscores['fine_P'].append(f_precision)
        # Fscores['fine_R'].append(f_recall)


        # Trainer
        trainer = DualRefineTrainer(model, model_spread, args, num_cluster=args.num_clusters)

        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, clean_labels, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader_target))

        def save_model(model, is_best, best_mAP):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            R1, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_model(model, is_best, best_mAP)

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

            # Fscores['map'].append(mAP)
            # Fscores['r1'].append(R1)

        # joblib.dump(Fscores, osp.join(args.logs_dir, 'fscore.pkl'))

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint Label and Feature Refinement")

    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--epochs_decay', type=int, default=20)

    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--rho', type=float, default=1.6e-3)
    parser.add_argument('--rr-gpu', action='store_true',
                        help="use GPU for accelerating clustering")

    # hyper-parameters setting
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='parameter to balance noisy&reliable losses')
    parser.add_argument('--lmd', type=float, default=0.5,
                        help='lmd for cross-entropy loss')
    parser.add_argument('--mu', type=float, default=0.1,
                        help='mu for spread loss')
    parser.add_argument('--knn', type=int, default=6,
                        help='number of k-nearest neighborhoods in spread loss')
    parser.add_argument('--m', type=float, default=0.35,
                        help='margin for spread loss')
    parser.add_argument('--R', type=int, default=2,
                        help='number of sub-clusters in off-line hierarchical clustering')

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
