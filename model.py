import keras.applications.inception_v3 as kai
import keras.layers as kl
import keras.models as km


def triplet_loss_all():
    
    def loss(y_true, y_pred):
        pass
    
    return loss


def build_model(image_shape=(299, 299, 3), embedding_length=128):
    backbone = kai.InceptionV3(input_shape=image_shape, include_top=False)
    
    x = kl.GlobalMaxPooling2D()(backbone.output)
    x = kl.Dense(embedding_length * 4)(x)
    x = kl.Dense(embedding_length * 2)(x)
    embedding = kl.Dense(embedding_length, name='embedding')(x)
    
    model = km.Model(inputs=[backbone.input], outputs=[embedding])
    model.compile('sgd', loss=triplet_loss_all())
    
    return model

