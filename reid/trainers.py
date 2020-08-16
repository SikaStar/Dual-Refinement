from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter
import numpy as np


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec


class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500, lmd=1.0):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster
        self.lmd = lmd
        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t = self.model(inputs, feature_withbn=False)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = self.lmd*loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


class DualRefineTrainer(object):
    def __init__(self, model, model_spread, args, num_cluster=500):
        super(DualRefineTrainer, self).__init__()
        self.model = model
        self.model_spread = model_spread
        self.num_cluster = num_cluster

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

        self.lmd = args.lmd # parameter for cls_loss
        self.alpha = args.alpha  # balance noisy&reliable losses
        self.mu = args.mu # parameter for loss_spread



    def train(self, epoch, data_loader_target, clean_labels, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_clean_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_clean_tri = AverageMeter()
        precisions = AverageMeter()
        losses_spread = AverageMeter()
        losses = AverageMeter()

        clean_labels = clean_labels.cuda()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexs = self._parse_data(target_inputs)
            new_labels = clean_labels[indexs]

            # forward
            f_out_t, p_out_t = self.model(inputs, feature_withbn=True)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, labels)
            loss_clean_ce = self.criterion_ce(p_out_t, new_labels)

            loss_tri = self.criterion_tri(f_out_t, f_out_t, labels)
            loss_clean_tri = self.criterion_tri(f_out_t, f_out_t, new_labels)

            loss_spread = self.model_spread(f_out_t, indexs, epoch=epoch)

            loss = (1 - self.alpha) * (self.lmd * loss_ce + loss_tri) + \
                    self.alpha * (self.lmd * loss_clean_ce + loss_clean_tri) + \
                    self.mu * loss_spread

            optimizer.zero_grad()
            loss.backward()
            # multiple (1./lmd) in order to remove the effect of lmd on updating centers
            if self.lmd>0:
                for param in self.model_spread.parameters():
                    param.grad.data *= (1. / self.lmd)

            optimizer.step()

            prec, = accuracy(p_out_t.data, labels.data)

            losses_ce.update(loss_ce.item())
            losses_clean_ce.update(loss_clean_ce.item())
            losses_tri.update(loss_tri.item())
            losses_clean_tri.update(loss_clean_tri.item())
            losses_spread.update(loss_spread.item())
            losses.update(loss.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})  '
                      'Data {:.3f} ({:.3f})  '
                      'Loss {:.3f} ({:.3f})  '
                      'noisy_ce {:.3f} ({:.3f})  '
                      'clean_ce {:.3f} ({:.3f})  '
                      'noisy_tri {:.3f} ({:.3f})  '
                      'clean_tri {:.3f} ({:.3f})  '
                      'spread {:.3f} ({:.3f})  '
                      'Prec {:.2%} ({:.2%})  '
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_clean_ce.val, losses_clean_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_clean_tri.val, losses_clean_tri.avg,
                              losses_spread.val, losses_spread.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexs = inputs
        inputs = imgs.cuda()
        labels = pids.cuda()
        indexs = indexs.cuda()
        return inputs, labels, indexs
    