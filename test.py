import keras.utils as ku
import matplotlib.pyplot as plt

from data import BeeLoader
from model import build_model


if __name__ == '__main__':
    bee_data = BeeLoader()
    for i, (images, ids) in enumerate(bee_data.generator(6)):
        print('images shape: ', images.shape)
        print('ids: ', ids)
        plt.imshow(images[0])
        plt.show()
        break
    
    bee_model = build_model()
    with open('model_summary.txt', 'w') as f:
        ku.print_summary(bee_model, print_fn=lambda string: f.write(f'{string}\n'), line_length=200)
