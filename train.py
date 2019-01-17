import logging
import torch
from torch.autograd import Variable
from fastprogress import master_bar, progress_bar

from loss import f1_loss, acc
from evaluate import evaluate
import utils


def train_model(model, dataloader, loss_fn, optimizer, scheduler, params):

    # convert model to train, avoid model state is eval
    model.train()

    total = len(dataloader)



    train_loss = 0.0
    train_acc = 0.0

    for j in progress_bar(range(total), parent=params.mb):
        running_loss = 0.0
        # running_f1_loss = 0.0
        acc_rate = 0.0
        
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    async=True), labels_batch.cuda(async=True)

            # convert to torch Variables
            train_batch, labels_batch = Variable(
                train_batch), Variable(labels_batch)

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # adjust learnign rate
            scheduler.step()

            running_loss += loss.item()
            # temp_f1_loss = f1_loss(output_batch, labels_batch)
            # running_f1_loss += temp_f1_loss
            temp_acc_rate = acc(output_batch, labels_batch)
            acc_rate += temp_acc_rate

            params.mb.child.comment = "loss {} and acc {}".format(loss.item(), temp_acc_rate)
        
        train_loss = running_loss / total
        train_acc = acc_rate / total



    return train_loss, train_acc


def train_and_evaluate(model, dataloaders, optimizer, loss_fn, scheduler, params):

    metrics = utils.RunningMetrics()

    best_val_acc = 0.0

    mb = master_bar(range(params.num_epochs))

    params.mb = mb

    for epoch in mb:
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train_loss, train_acc = train_model(
            model, dataloaders["train"], loss_fn, optimizer, scheduler, params)
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], loss_fn, params)

        list = []
        list.append(train_loss)
        list.append(train_acc)
        list.append(val_loss)
        list.append(val_acc)
        metrics.update(list)

        logging.info("Epoch {}/{}. Train loss: {}, Train acc: {}. Val loss: {}, Val acc: {}".format(
            epoch + 1, params.num_epochs, train_loss, train_acc, val_loss, val_acc))

        is_best = val_acc >= best_val_acc

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=params.model_dir)

        if is_best:
            logging.info("- Found new best val acc")

            best_val_acc = val_acc

    metrics.save_to_csv(params.model_metrics_file)
