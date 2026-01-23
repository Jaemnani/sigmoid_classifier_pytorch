
import os
import shutil as sh
import torch
from glob import glob

class CheckpointManager:
    def __init__(self):
        self.model_name = None
        self.checkpoint_path = None
        self.best_metric = None

    def set_model_name(self, model_name):
        self.model_name = model_name

    def parse_pretrained_iteration_count(self, pretrained_model_path):
        iteration_count = 0
        sp = f'{os.path.basename(pretrained_model_path)[:-3]}'.split('_')
        for i in range(len(sp)):
            if sp[i] == 'iter' and i > 0:
                try:
                    iteration_count = int(sp[i-1])
                except:
                    pass
                break
        return iteration_count

    def make_checkpoint_dir(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def init_checkpoint_dir(self):
        inc = 0
        while True:
            if inc == 0:
                new_checkpoint_path = f'results/train/{self.model_name}'
            else:
                new_checkpoint_path = f'results/train/{self.model_name}_{inc}'
            if os.path.exists(new_checkpoint_path) and os.path.isdir(new_checkpoint_path):
                inc += 1
            else:
                break
        self.checkpoint_path = new_checkpoint_path
        self.make_checkpoint_dir()
        print(f'checkpoint path : {self.checkpoint_path}')
        return self.checkpoint_path

    def remove_last_model(self):
        for last_model_path in glob(f'{self.checkpoint_path}/last_*.pt'):
            os.remove(last_model_path)

    def save_last_model(self, state_dict, iteration_count, content=''):
        self.make_checkpoint_dir()
        save_path = f'{self.checkpoint_path}/last_{iteration_count}_iter{content}.pt'
        torch.save(state_dict, save_path)
        # Backup logic
        backup_path = f'{save_path}.bak'
        sh.copy(save_path, backup_path)
        self.remove_last_model()
        sh.move(backup_path, save_path)
        return save_path

    def remove_best_model(self):
        for best_model_path in glob(f'{self.checkpoint_path}/best_*.pt'):
            os.remove(best_model_path)

    def save_best_model(self, state_dict, iteration_count, metric, content=''):
        save_path = None
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.make_checkpoint_dir()
            save_path = f'{self.checkpoint_path}/best_{iteration_count}_iter{content}.pt'
            torch.save(state_dict, save_path)
            
            backup_path = f'{save_path}.bak'
            sh.copy(save_path, backup_path)
            self.remove_best_model()
            sh.move(backup_path, save_path)
        return save_path
