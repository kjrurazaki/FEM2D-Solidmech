import tensorflow as tf
from keras import backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
from keras.models import load_model
from matplotlib import pyplot as plt


class NN_model:
    """
    Class that stores all NNs configuration and have a method to compile the NN
    """

    def __init__(self, input_size, output_size):
        self.model_name = ""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1_size = 128
        self.hidden1_dropout = 0.2
        self.hidden1_activation = "relu"
        self.n_hiddenlayers = 2
        self.hidden2_size = 128
        self.hidden2_dropout = 0.2
        self.hidden2_activation = "relu"
        self.output_activation = "relu"
        self.learning_rate = 0.001
        self.loss = "mse"
        self.metrics = ["mae", "mse", coeff_determination]

    def build_NN(self):
        """
        Compile the model, uses RMSprop as optimizer and maximum 2 hidden layers
        """
        # Model with first hidden layer and dropout
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    self.hidden1_size,
                    activation=self.hidden1_activation,
                    input_shape=(self.input_size,),
                ),
                tf.keras.layers.Dropout(self.hidden1_dropout),
            ]
        )
        # Model with second hidden layer if chosen
        if self.n_hiddenlayers == 2:
            model.add(tf.keras.layers.Dense(self.hidden2_size)),
            model.add(tf.keras.layers.Dropout(self.hidden2_dropout))
        # output layer
        model.add(
            tf.keras.layers.Dense(self.output_size, activation=self.output_activation)
        )
        # Compile
        sgd = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        model.compile(loss=self.loss, optimizer=sgd, metrics=self.metrics)
        return model

    def update_NN(self, dict_update):
        self.model_name = dict_update["name"]
        self.hidden1_size = dict_update["hidden1_size"]
        self.hidden1_dropout = dict_update["hidden1_dropout"]
        self.n_hiddenlayers = dict_update["n_hiddenlayers"]
        self.hidden2_size = dict_update["hidden2_size"]
        self.hidden2_dropout = dict_update["hidden2_dropout"]
        self.learning_rate = dict_update["learning_rate"]


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def start_NN(data, builder_model, dict_NN, CV=True):
    """
    Call model compiler, defines keras callbacks and train model
    """
    X_train, X_test, Y_train, Y_test, scaler_input, scaler_output = data
    builder_model.update_NN(dict_NN)
    model = builder_model.build_NN()
    model.summary()
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
    mc = ModelCheckpoint(
        "best_model_" + builder_model.model_name + ".h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )
    lg = CSVLogger("history_" + builder_model.model_name + ".csv")
    lr_onplateu = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        verbose=1,
        min_delta=1e-6,
        min_lr=0.0001,
    )
    if CV == True:
        train_idx = dict_NN["train_idx"]
        val_idx = dict_NN["val_idx"]
        # Train and keep history
        history = model.fit(
            X_train[train_idx, :],
            Y_train[train_idx, :],
            validation_data=(X_train[val_idx, :], Y_train[val_idx, :]),
            epochs=50,
            batch_size=dict_NN["batch_size"],
            callbacks=[es, mc, lg, lr_onplateu],
            verbose=1,
        )
    else:
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_train, Y_train),
            epochs=50,
            batch_size=dict_NN["batch_size"],
            callbacks=[es, mc, lg, lr_onplateu],
            verbose=1,
        )
    return history


def test_error(data, name, scaled_target=True):
    """
    Test model 'name' saved for test set of data
    """
    X_train, X_test, Y_train, Y_test, scaler_input, scaler_output = data
    saved_model = load_model(
        "best_model_" + name + ".h5",
        custom_objects={"coeff_determination": coeff_determination},
    )
    Y_pred = saved_model.predict(X_test)
    if scaled_target == True:
        Y_pred = scaler_output.inverse_transform(Y_pred)
        Y_test = scaler_output.inverse_transform(Y_test)
        print(saved_model.metrics_names)
        print(saved_model.evaluate(X_test, Y_test))
    plt.scatter(Y_pred, Y_test)
    plt.plot([Y_pred.min(), Y_pred.max()], [Y_test.min(), Y_test.max()], "--")
