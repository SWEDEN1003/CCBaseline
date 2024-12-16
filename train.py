import datetime
import logging
import time

import torch
from torch import nn
from torch.nn import MSELoss
from torchvision import transforms as T

from losses.part_based_matching_loss import match_loss
from tools.utils import AverageMeter

torch.set_printoptions(profile="full")


def train_model(
    config,
    epoch,
    model,
    criterion_cla,
    criterion_pair,
    optimizer,
    trainloader,
):
    logger = logging.getLogger("reid.train")
    batch_cla_loss_f = AverageMeter()
    batch_pair_loss_f = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.train()
    # Resize = T.Resize((24, 12))
    end = time.time()
    Total_iteration = len(trainloader)

    for batch_idx, (parsing_results, imgs, imgs_b, pids, clothes_ids) in enumerate(
        trainloader
    ):

        parsing_results, imgs, imgs_b, pids, clothes_ids = (
            parsing_results.cuda(),
            imgs.cuda(),
            imgs_b.cuda(),
            pids.cuda(),
            clothes_ids.cuda(),
        )
        data_time.update(time.time() - end)
        score, global_feat = model(imgs)
        # parsing_results = Resize(parsing_results)
        pair_loss_f = criterion_pair(global_feat, pids)
        cla_loss_f = criterion_cla(score, pids)
        _, preds = torch.max(score.data, 1)
        # Please adjust the weight of loss functions according to different datasets
        loss = pair_loss_f + cla_loss_f
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(
            torch.sum(preds == pids.data).float() / pids.size(0), pids.size(0)
        )
        batch_pair_loss_f.update(pair_loss_f.item(), pids.size(0))
        batch_cla_loss_f.update(cla_loss_f.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                "Epoch{0} Iteration[{1}/{2}] "
                "Tri loss:{pair_loss_f.avg:.4f} "
                "Cls loss:{cla_loss_f.avg:.4f} "
                "Acc:{acc.avg:.2%} "
                "Time:{batch_time.avg:.1f}s "
                "Data:{data_time.avg:.1f}s ".format(
                    epoch + 1,
                    batch_idx + 1,
                    Total_iteration + 1,
                    cla_loss_f=batch_cla_loss_f,
                    pair_loss_f=batch_pair_loss_f,
                    acc=corrects,
                    batch_time=batch_time,
                    data_time=data_time,
                )
            )

    logger.info(
        "Epoch{0} "
        "Time:{batch_time.sum:.1f}s "
        "Data:{data_time.sum:.1f}s "
        "Tri loss:{pair_loss_f.avg:.4f} "
        "Cls loss:{cla_loss_f.avg:.4f} "
        "Acc:{acc.avg:.2%} ".format(
            epoch + 1,
            batch_time=batch_time,
            data_time=data_time,
            cla_loss_f=batch_cla_loss_f,
            pair_loss_f=batch_pair_loss_f,
            acc=corrects,
        )
    )
