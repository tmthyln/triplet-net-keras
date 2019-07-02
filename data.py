import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.transform


BEE_TYPES = ['bumble', 'carpenter', 'honey', 'mason', 'mining', 'yellowface']


class BeeLoader(object):
    
    def __init__(self, path='./data'):
        if not os.path.exists(path):
            raise ValueError('provided path to data must exist')
        
        _, _, files = next(os.walk(path))
        
        self.root = path
        self.by_type = {bee_type: [file for file in files if file.startswith(bee_type)] for bee_type in BEE_TYPES}
        self.all = files
    
    def load_batch_from_files(self, filenames):
        images = []
        
        for file in filenames:
            images.append(self.load_resized_image(os.path.join(self.root, file)))
            
        return np.stack(images, axis=0)

    @staticmethod
    def load_resized_image(filename, size=299):
        raw = plt.imread(filename)

        curr_crop = skimage.transform.rescale(raw, size / max(len(raw), len(raw[0])),
                                              mode='constant', multichannel=True)
        return np.pad(curr_crop, ((0, size - len(curr_crop)), (0, size - len(curr_crop[0])), (0, 0)),
                      mode='constant')

    def generator(self, batch_size):
        
        assert batch_size >= 3, 'batch size must be at least 3 (enough for at least 1 triplet'
        
        pos_pair_index = 0
        while True:
            filenames_in_batch = []
            types_in_batch = []
            
            # guaranteed positive pair
            pos_type = BEE_TYPES[pos_pair_index]
            for i in range(2):
                filenames_in_batch.append(self.by_type[pos_type][np.random.choice(len(self.by_type[pos_type]))])
                types_in_batch.append(pos_pair_index)
            
            # fill the rest of the batch
            for i in np.random.choice(len(self.all), batch_size - 2, replace=False):
                filenames_in_batch.append(self.all[i])
                types_in_batch.append(BEE_TYPES.index(''.join(itertools.takewhile(str.isalpha, self.all[i]))))
            
            # load images from file names, return (x, y)
            yield self.load_batch_from_files(filenames_in_batch), np.array(types_in_batch)
            
            # change the species we're using for the positive pair
            pos_pair_index = (pos_pair_index + 1) % len(BEE_TYPES)
