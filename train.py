
import os
import random
from functools import partial
from glob import glob
from os.path import join, basename
import pickle
from transformers import BeitForImageClassification

from tqdm import tqdm
from garbage_data import GarbageData
from config import config as opt
from sklearn.metrics import f1_score as measure_f1_score
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

writer = SummaryWriter(log_dir=os.path.join(opt.checkpoint_dir, 'tf_log'))

def score_model(model, dataloader):
    """
    :param model:
    :param dataloader:
    :return:
        res: f1_score
    """
    print('Model scoring was started')
    model.eval()
    dataloader.dataset.mode = 'eval'
    result = []
    targets = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            frames = torch.squeeze(batch[0]['pixel_values']).to(opt.device)
            labels = batch[1].to(opt.device)
            predicted = model(frames)
            predicted = predicted.logits.argmax(dim=-1)
            result.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    f1_score = measure_f1_score(targets, result, average='macro')
    return f1_score

def train_epoch(epoch, model, dataloader, optimizer):
    print(f'start {epoch} epoch')
    for step, batch in tqdm(enumerate(dataloader)):
        frames = torch.squeeze(batch[0]['pixel_values']).to(opt.device)
        labels = batch[1].to(opt.device)
        predicted = model(frames)
        loss = F.cross_entropy(predicted.logits, labels)
        writer.add_scalar('loss', loss.item(), global_step=epoch * len(dataloader) + step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_model():
    garbage_data = GarbageData(opt.dataset_dir, opt.batch_size, opt.val_part)
    train_dataloader = garbage_data.get_train_loader()
    val_dataloader = garbage_data.get_val_loader()
    if opt.load_checkpoint_path:
        model = torch.load(opt.load_checkpoint_path)
        print(f'model was loaded from {opt.load_checkpoint_path}')
    else:
        model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        model.classifier = torch.nn.Linear(768, opt.classes)
        model.config.num_labels = opt.classes
    
    model.to(opt.device)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    for epoch in range(opt.n_epoch):
        model = train_epoch(epoch, model, train_dataloader, optimizer)
        torch.save(model, os.path.join(opt.checkpoint_dir, 'weights/latest/latest.pth'))
        if epoch % opt.scoring_everyN_epoch == 0:
            f1_score = score_model(model, val_dataloader)
            writer.add_scalar('val_f1', f1_score, (epoch + 1) * len(train_dataloader))
            torch.save(model, join(opt.checkpoint_dir, f'epoch{epoch}_f1={round(f1_score, 5)}.pth'))
    return model

def main():
    model = train_model()
        
if __name__ == '__main__':
    main()