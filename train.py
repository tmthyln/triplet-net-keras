print(len(sys.argv))
print("hello world")

import keras.utils as ku
import matplotlib.pyplot as plt

from data import BeeLoader
from model import build_model

bee_data = BeeLoader()
bee_model = build_model()

for images, ids in bee_data.generator(6):
  pass
  #bee_model.train([images], [id])
