# for quick exacting feature in format in README.md
# copy this file to your model folder, use it to extract and save feature

import numpy as np
import pickle
import os

class save_helper:
    def __init__(self, save_dir='data') -> None:
        self.save_dir = save_dir
        self.features = []
        self.pic_names = []
    
    def add_feature(self, feature):
        pass
    
    def add_name(self, info):
        pass
    
    def save(self):
        if os.path.exists(self.save_dir):
            os.rmdir(self.save_dir)
        os.mkdir(self.save_dir)
        
        