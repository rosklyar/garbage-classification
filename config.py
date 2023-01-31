import os
class Config:
    def __init__(self):
        self.dataset_dir = 'garbage_dataset'
        self.checkpoint_dir = 'checkpoints/garbage_classification_nn'
        self.device = 'cuda:0'
        self.load_checkpoint_path = None
        # number of subfolders in dataset_dir
        self.classes = len(os.listdir(self.dataset_dir))
        self.batch_size = 16
        self.val_part = 0.2
        self.n_epoch = 40
        self.lr = 0.0002
        self.scoring_everyN_epoch = 3

config = Config()
os.makedirs(os.path.join(config.checkpoint_dir, 'weights/latest'), exist_ok=True)