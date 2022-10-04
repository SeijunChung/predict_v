import io
import pdb
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
import logging
import matplotlib.pyplot as plt
from model.model_conv import conv1d_mlp, conv1d_att
from datetime import datetime
import os
import pathlib
import shutil
import argparse

# For pycharm autocomplete issue
import typing
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as tk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class prediction():
    def __init__(self,
                 batch_size=20,
                 epochs=10,
                 patience=20,
                 lr=5e-4,
                 lr_decay=None,
                 log_freq=1000,
                 multi_gpu=True,
                 past_windowsize=20,
                 past_stepsize=5,
                 pred_windowsize=10,
                 pred_stepsize=10,
                 past_filters=[32, 32, 32, 1],  # for past encoder
                 past_sizes=[3, 3, 3, 1],  # for past encoder kernels
                 past_strides=[1, 1, 1, 1],  # for past encoder
                 forward_filters=[32, 32, 32, 1],  # for forward encoder
                 forward_sizes=[5, 5, 5, 1],  # for forward encoder kernels
                 forward_strides=[2, 2, 2, 1],  # for forward encoder
                 forward_dilation_rates=[],
                 results_path='./results',
                 run_test=False):
        # Parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch = 0
        self.patience = patience
        self.lr = lr
        self.lr_decay = lr_decay
        self.log_freg = log_freq
        self.past_windowsize = past_windowsize
        self.past_stepsize = past_stepsize
        self.pred_windowsize = pred_windowsize
        self.pred_stepsize = pred_stepsize
        self.run_test = run_test
        self.multi_gpu = multi_gpu

        if self.run_test:
            self.bpe_train = 1
            self.bpe_valid = 1
            self.bpe_test = 1
            self.epochs = 1
            self.batch_size = 10

        if self.multi_gpu:
            self.strategy = tf.distribute.MirroredStrategy()
            self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
        else:
            self.global_batch_size = self.batch_size

        # set logger
        self.mylogger = logging.getLogger("my")
        self.mylogger.propagate = False
        self.mylogger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.mylogger.addHandler(stream_handler)

        # Read data from pickle file
        self.current_folder_path = str(pathlib.Path(__file__).parent.parent.resolve()) + '/'
        self.file_prefix = self.current_folder_path + 'data/hatci_clinic/1st_TD_matched/dataset_1/'

        # results path
        self.starttime = datetime.now()
        self.results_path = self.current_folder_path + results_path + '/' + 'att_' + self.starttime.strftime("%Y%m%d-%H%M%S")

        # TensorBoard
        train_log_dir = self.results_path + '/train'
        valid_log_dir = self.results_path + '/valid'
        img_log_dir = self.results_path + '/image'
        self.train_tb_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_tb_writer = tf.summary.create_file_writer(valid_log_dir)
        self.train_tbimg_writer = tf.summary.create_file_writer(img_log_dir + '/train')
        self.valid_tbimg_writer = tf.summary.create_file_writer(img_log_dir + '/valid')

        # train and model python file copy to results directory
        shutil.copy('./exp_att.py', self.results_path + '/exp_att.py')
        shutil.copy('./model/model_conv.py', self.results_path + '/model_conv.py')

        # save experiment parameters
        with open(self.results_path + '/hyperparams.txt', 'w') as f:
            f.write(f'dataset:\t{self.file_prefix}\n')
            f.write(f'learning_rate:\t{self.lr}\n')
            f.write(f'batch_size:\t{self.global_batch_size}\n')
            f.write(f'epochs:\t{self.epochs}\n')
            f.write(f'multi_gpu:\t{self.multi_gpu}\n')
            f.write(f'past_windowsize:\t{self.past_windowsize}\n')
            f.write(f'past_stepsize:\t{self.past_stepsize}\n')
            f.write(f'pred_windowsize:\t{self.pred_windowsize}\n')
            f.write(f'pred_stepsize:\t{self.pred_stepsize}\n')

        # model, optimizer and metric
        if self.multi_gpu:
            with self.strategy.scope():
                self.model = conv1d_att(past_filters, past_sizes, past_strides, forward_filters, forward_sizes, forward_strides)
                self.optimizer = tk.optimizers.Adam(lr)
                self.loss_objective = tk.losses.MeanSquaredError(reduction=tk.losses.Reduction.NONE)
                self.mae_objective = tk.losses.MeanAbsoluteError(reduction=tk.losses.Reduction.NONE)
                self.train_metric_mae = tk.metrics.Mean('train_mae')
                self.valid_metric_mae = tk.metrics.Mean('valid_mae')
            self.test_loss = tk.metrics.Mean('test_loss')
            self.train_loss = tk.metrics.Mean('train_loss')
            self.valid_loss = tk.metrics.Mean('valid_loss')
        else:
            self.model = conv1d_att(past_filters, past_sizes, past_strides, forward_filters, forward_sizes, forward_strides)
            self.optimizer = tk.optimizers.Adam(lr)
            self.loss_objective = tk.losses.MeanSquaredError(reduction=tk.losses.Reduction.NONE)
            self.mae_objective = tk.losses.MeanAbsoluteError(reduction=tk.losses.Reduction.NONE)
            self.train_loss = tk.metrics.Mean('train_loss')
            self.train_metric_mae = tk.metrics.Mean('train_mae')
            self.valid_loss = tk.metrics.Mean('valid_loss')
            self.valid_metric_mae = tk.metrics.Mean('valid_mae')
            self.test_loss = tk.metrics.Mean('test_loss')

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        # plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def result_plot(self, x_batch, y_batch, outputs):
        figure = plt.figure()
        for i in range(10):
            plt.clf()
            plt.plot((np.arange(0, self.past_stepsize * self.past_windowsize, self.past_stepsize)
                      - self.past_stepsize * (self.past_windowsize - 1)) * 0.1,
                     self.pred_max * x_batch[i, :, -1])
            plt.plot((np.arange(self.past_stepsize * (self.past_windowsize - 1) + self.pred_stepsize,
                                self.past_stepsize * (
                                            self.past_windowsize - 1) + self.pred_stepsize * self.pred_windowsize + 1,
                                self.pred_stepsize) - self.past_stepsize * (self.past_windowsize - 1)) * 0.1,
                     self.pred_max * y_batch[i, :, -1])
            plt.plot((np.arange(self.past_stepsize * (self.past_windowsize - 1) + self.pred_stepsize,
                                self.past_stepsize * (
                                            self.past_windowsize - 1) + self.pred_stepsize * self.pred_windowsize + 1,
                                self.pred_stepsize) - self.past_stepsize * (self.past_windowsize - 1)) * 0.1,
                     self.pred_max * outputs[i, :, -1])
            plt.grid()
            plt.tight_layout(pad=0)
            plt.legend(['past', 'truth', 'pred'])

            image = self.plot_to_image(figure)
            if i == 0:
                images = image
            else:
                images = tf.concat([images, image], axis=0)
        plt.close(figure)

        return images

    def data_normalizer(self, data):
        max_value = np.max(np.abs(data[:, 0, :]), axis=0)

        return max_value

    def data_loader(self, phase):
        # past data column: ['pv', 'aglSteering', 'prBrakeMasterCylinder', 'distObjectAcc', 'spdRelObjectAcc', 'vsTcu']
        data = np.load(self.file_prefix + 'dataset_past_' + str(phase) + '.npy', mmap_mode='r')
        past = data[:,  -self.past_stepsize*(self.past_windowsize-1)-1::self.past_stepsize]
        past = past.astype('float32')
        map_data = np.load(self.file_prefix + 'dataset_map_' + str(phase) + '.npy')
        map_data = map_data.astype('float32')
        pred = np.load(self.file_prefix + 'dataset_pred_' + str(phase) + '.npy')
        if self.pred_stepsize == 10:
            pred = pred[:, 1::2]
        pred = pred.astype('float32')

        return past, map_data, pred

    def pos_encoder(self, pos, freq=400, d=4):
        pos_enc = np.zeros([pos.shape[0], d], dtype=np.float32)
        for i in range(d):
            if i % 2 == 0:
                pos_enc[:, i] = np.sin(pos / (freq ** (i / d)))
            else:
                pos_enc[:, i] = np.cos(pos / (freq ** (i / d)))
        return pos_enc

    @tf.function(jit_compile=False)
    def train_step(self, dataset_inputs):
        print('retracing')
        past_batch, map_batch, y_batch = dataset_inputs
        with tf.GradientTape() as tape:
            outputs = self.model(past_batch, map_batch, self.pos_enc_past, self.pos_enc_fwd, training=True)
            loss = tf.nn.compute_average_loss(tf.reduce_mean(self.loss_objective(y_batch, outputs), axis=-1),
                                              global_batch_size=self.global_batch_size)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_metric_mae.update_state(
            tf.reduce_mean(self.mae_objective(y_batch * self.pred_max, outputs * self.pred_max), axis=-1))

        return loss, past_batch, outputs, y_batch

    @tf.function(jit_compile=False)
    def valid_step(self, dataset_inputs):
        past_batch, map_batch, y_batch = dataset_inputs
        outputs = self.model(past_batch, map_batch, self.pos_enc_past, self.pos_enc_fwd, training=False)
        loss = tf.nn.compute_average_loss(tf.reduce_mean(self.loss_objective(y_batch, outputs), axis=-1),
                                          global_batch_size=self.global_batch_size)

        self.valid_metric_mae.update_state(
            tf.reduce_mean(self.mae_objective(y_batch * self.pred_max, outputs * self.pred_max), axis=-1))

        return loss, past_batch, outputs, y_batch

    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses, past_batch, outputs, y_batch = self.strategy.run(self.train_step, args=(dataset_inputs,))

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), past_batch.values[0], \
               outputs.values[0], y_batch.values[0]

    @tf.function
    def distributed_valid_step(self, dataset_inputs):
        per_replica_losses, past_batch, outputs, y_batch = self.strategy.run(self.valid_step, args=(dataset_inputs,))

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None), past_batch.values[0], \
               outputs.values[0], y_batch.values[0]

    def train(self):
        # dataset loading
        train_past, train_map, train_pred = self.data_loader('train')
        valid_past, valid_map, valid_pred = self.data_loader('valid')
        self.pos_enc_past = tf.tile(tf.expand_dims(self.pos_encoder(np.arange(10), 40), axis=0), [self.batch_size, 1, 1])
        self.pos_enc_fwd = tf.tile(tf.expand_dims(self.pos_encoder(np.arange(100)), axis=0), [self.batch_size, 1, 1])

        # Model initialization
        with self.strategy.scope():
            self.model(train_past[:1], train_map[:1], self.pos_enc_past[:1], self.pos_enc_fwd[:1])
        with self.train_tb_writer.as_default():
            for weights in self.model.trainable_weights:
                tf.summary.histogram(weights.name, weights, step=0)

        # dataset normalization (max)
        self.past_max = self.data_normalizer(train_past)
        self.map_max = self.data_normalizer(train_map)
        self.pred_max = self.past_max[-1:]
        train_past = train_past / self.past_max
        train_map = train_map/self.map_max
        train_pred = train_pred / self.pred_max
        valid_past = valid_past / self.past_max
        valid_map = valid_map/self.map_max
        valid_pred = valid_pred / self.pred_max

        # tf.dataset preparation
        if self.run_test:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_past[:self.global_batch_size], train_map[:self.global_batch_size],
                 train_pred[:self.global_batch_size])).batch(
                self.global_batch_size)
            valid_dataset = tf.data.Dataset.from_tensor_slices(
                (valid_past[:self.global_batch_size], valid_map[:self.global_batch_size],
                 valid_pred[:self.global_batch_size])).batch(
                self.global_batch_size)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_past, train_map, train_pred)).shuffle(
                train_past.shape[0], reshuffle_each_iteration=True).batch(
                self.global_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2).repeat(
                self.epochs)
            valid_dataset = tf.data.Dataset.from_tensor_slices((valid_past, valid_map, valid_pred)).shuffle(
                valid_past.shape[0]).batch(
                self.global_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2).repeat(
                self.epochs)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 0
            train_dataset = train_dataset.with_options(options)

        if self.multi_gpu:
            train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
            valid_dist_dataset = self.strategy.experimental_distribute_dataset(valid_dataset)
            trainer = self.distributed_train_step
            valider = self.distributed_valid_step
        else:
            train_dist_dataset = train_dataset
            valid_dist_dataset = valid_dataset
            trainer = self.train_step
            valider = self.valid_step

        train_dist_dataset_iterator = iter(train_dist_dataset)
        valid_dist_dataset_iterator = iter(valid_dist_dataset)
        if not self.run_test:
            self.bpe_train = int(np.floor(train_past.shape[0] / self.global_batch_size))  # bpe : batch per epoch
            self.bpe_valid = int(np.floor(valid_past.shape[0] / self.global_batch_size))

        # learning rate schedule
        if self.lr_decay:
            self.lr_schedule = tk.optimizers.schedules.CosineDecayRestarts(self.lr,
                                                                          first_decay_steps=20*self.bpe_train,
                                                                          t_mul=1.5,
                                                                          m_mul=0.98,
                                                                          alpha=0.1*self.lr)
            self.optimizer.learning_rate = self.lr_schedule
        
        # callbacks
        callbacks = tf.keras.callbacks.CallbackList()
        earlystopper = tk.callbacks.EarlyStopping(monitor='valid_loss', patience=self.patience, verbose=1, mode='min',
                                                  restore_best_weights=True)
        callbacks.append(earlystopper)
        callbacks.set_model(model=self.model)

        # training loop
        callbacks.on_train_begin()
        for epoch in range(self.epochs):
            self.mylogger.info(f"Training epoch {epoch + 1} / {self.epochs}")
            for step in range(self.bpe_train):
                loss, past_batch, outputs, y_batch = trainer(train_dist_dataset_iterator.get_next())
                self.train_loss.update_state(loss)

                if (step + 1) == self.bpe_train:
                    self.mylogger.info("step %d / %d" % (step + 1, self.bpe_train))
                    self.mylogger.info("\tTrain Loss = %.4f" % (loss * self.pred_max ** 2))
                    self.mylogger.info("\tTrain MAE = %.4f" % (self.train_metric_mae.result()))

            # TensorBoard write
            with self.train_tb_writer.as_default():
                tf.summary.scalar("MSE Loss", self.train_loss.result(), step=epoch + 1)
                tf.summary.scalar("MAE", self.train_metric_mae.result(), step=epoch + 1)
                for weights in self.model.trainable_weights:
                    tf.summary.histogram(weights.name, weights, step=epoch + 1)

            images = self.result_plot(past_batch, y_batch, outputs)
            with self.train_tbimg_writer.as_default():
                tf.summary.image("Train prediction", images, max_outputs=25, step=epoch + 1)

            # Validation
            for step in range(self.bpe_valid):
                loss, past_batch, outputs, y_batch = valider(valid_dist_dataset_iterator.get_next())
                self.valid_loss.update_state(loss)

            self.mylogger.info("\tValid Loss = %.4f" % (self.valid_loss.result() * self.pred_max ** 2))
            self.mylogger.info("\tValid MAE = %.4f" % (self.valid_metric_mae.result()))

            # TensorBoard write valid
            with self.valid_tb_writer.as_default():
                tf.summary.scalar("MSE Loss", self.valid_loss.result(), step=epoch + 1)
                tf.summary.scalar("MAE", self.valid_metric_mae.result(), step=epoch + 1)

            images = self.result_plot(past_batch, y_batch, outputs)
            with self.valid_tbimg_writer.as_default():
                tf.summary.image("Valid prediction", images, max_outputs=25, step=epoch + 1)

            # check early stop
            callbacks.on_epoch_end(epoch, logs={'valid_loss': self.valid_loss.result()})
            if self.model.stop_training and epoch+1 > 100:
                self.epoch = epoch
                break

            # reset metric states
            self.train_loss.reset_states()
            self.train_metric_mae.reset_states()
            self.valid_loss.reset_states()
            self.valid_metric_mae.reset_states()

    def test(self):
        self.test_past, self.test_map, self.test_pred = self.data_loader('test')

        # dataset normalization
        self.test_past = self.test_past / self.past_max
        self.test_map = self.test_map/self.map_max
        self.test_pred = self.test_pred / self.pred_max

        outputs = np.zeros_like(self.test_pred)
        test_id = np.arange(self.test_past.shape[0])
        if not self.run_test:
            self.bpe_test = int(np.ceil(len(test_id) / self.batch_size))
        for step in range(self.bpe_test):
            if step == self.bpe_test - 1 and not self.run_test:
                tupids = test_id[step * self.batch_size:]
            else:
                tupids = test_id[step * self.batch_size:(step + 1) * self.batch_size]
            past_batch = self.test_past[tupids]
            map_batch = self.test_map[tupids]
            y_batch = self.test_pred[tupids]
            pos_enc_past = tf.tile(tf.expand_dims(self.pos_encoder(np.arange(10), 40), axis=0), [len(tupids), 1, 1])
            pos_enc_fwd = tf.tile(tf.expand_dims(self.pos_encoder(np.arange(100)), axis=0), [len(tupids), 1, 1])

            output = self.model(past_batch, map_batch, pos_enc_past, pos_enc_fwd, training=False)
            if step == self.bpe_test - 1 and not self.run_test:
                outputs[step * self.batch_size:] = output.numpy()
            else:
                outputs[step * self.batch_size:(step + 1) * self.batch_size] = output.numpy()
            test_loss = self.loss_objective(y_batch, output)

            self.test_loss.update_state(test_loss)
        self.mylogger.info("Test Results")
        self.mylogger.info("\tTest Loss = %.4f" % (self.test_loss.result() * self.pred_max ** 2))
        self.test_loss.reset_states()

        return outputs

    def evaluation(self):
        # run test dataset
        outputs = self.test()
        # save test output
        eval_path = self.results_path + '/evaluations'
        if not os.path.isdir(eval_path):
            os.mkdir(eval_path)
        np.save(eval_path + '/test_output', outputs)
        np.save(eval_path + '/past_max', self.past_max)
        np.save(eval_path + '/map_max', self.map_max)
        np.save(eval_path + '/pred_max', self.pred_max)
        # evaluation
        error = self.pred_max * (self.test_pred - outputs)
        mse = np.mean(np.square(error), axis=0)
        mae = np.mean(np.abs(error), axis=0)
        ss_tot = np.sum(np.square(self.test_pred - np.mean(self.test_pred, axis=0)), axis=0)
        ss_res = np.sum(np.square(self.test_pred - outputs), axis=0)
        r2 = 1 - ss_res / ss_tot
        # print evaluation results
        np.set_printoptions(precision=4, suppress=True)
        print("mse")
        print(mse)
        print("mean = ", np.mean(mse))
        print("mae")
        print(mae)
        print("mean = ", np.mean(mae))
        print("r2")
        print(r2)
        print("mean = ", np.mean(r2))
        # model summary
        self.model.summary(expand_nested=True)
        # save evaluation results
        # numpy
        np.save(eval_path + '/eval_mse', mse)
        np.save(eval_path + '/eval_mae', mae)
        np.save(eval_path + '/eval_r2', r2)
        # txt file
        with open(self.results_path + '/experiments_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f'Hyperparameters\n')
            f.write(f'dataset:\t{self.file_prefix}\n')
            f.write(f'learning_rate:\t{self.lr}\n')
            f.write(f'batch_size:\t{self.batch_size}\n')
            f.write(f'global_batch_size:\t{self.global_batch_size}\n')
            f.write(f'patience:\t{self.patience}\n')
            f.write(f'epochs:\t{self.epochs-self.patience+1}\n')
            f.write(f'multi_gpu:\t{self.multi_gpu}\n')
            f.write(f'past_windowsize:\t{self.past_windowsize}\n')
            f.write(f'past_stepsize:\t{self.past_stepsize}\n')
            f.write(f'pred_windowsize:\t{self.pred_windowsize}\n')
            f.write(f'pred_stepsize:\t{self.pred_stepsize}\n')
            f.write(f'finished at epoch:\t{self.epoch}\n')
            f.write('\n')
            f.write(f'mse:\n')
            f.write(f'{mse}\n')
            f.write(f'mean:\t{np.mean(mse)}\n')
            f.write(f'mae:\n')
            f.write(f'{mae}\n')
            f.write(f'mean:\t{np.mean(mae)}\n')
            f.write(f'r2:\n')
            f.write(f'{r2}\n')
            f.write(f'mean:\t{np.mean(r2)}\n')
            f.write('\n')
            f.write('Model Summary\n')
            self.model.summary(print_fn=lambda x: f.write(x + '\n'), expand_nested=True)

    def save_trained_model(self):
        self.model.save_weights(self.results_path + '/weights/weight')
        self.model.save(self.results_path + '/model_pb')
        model_json = self.model.to_json()
        with open(self.results_path + '/model.json', 'w') as f:
            f.write(model_json)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training and saving prediction model')
    parser.add_argument('--batch_size', type=int, action='store', default=2000, help='training batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='the number of epochs')
    parser.add_argument('--lr_decay', default=True, action='store_false', help='lr scheduling')
    parser.add_argument('--patience', type=int, default=20, help='the number of patience')
    parser.add_argument('--past_stepsize', type=int, action='store', default=1, help='past_stepsize')
    parser.add_argument('--past_windowsize', type=int, action='store', default=20, help='past_windowsize')
    parser.add_argument('--pred_stepsize', type=int, action='store', default=10, help='pred_stepsize')
    parser.add_argument('--pred_windowsize', type=int, action='store', default=10, help='pred_windowsize')
    args = parser.parse_args()

    vel_predictor = prediction(batch_size=args.batch_size,
                               lr=5e-4,
                               lr_decay=args.lr_decay,
                               epochs=args.epochs,
                               patience=args.patience,
                               log_freq=200,
                               multi_gpu=True,
                               past_stepsize=args.past_stepsize,
                               past_windowsize=args.past_windowsize,
                               pred_stepsize=args.pred_stepsize,
                               pred_windowsize=args.pred_windowsize,
                               past_filters=[32, 32, 32],
                               past_sizes=[3, 3, 3],
                               past_strides=[1, 1, 1],
                               forward_filters=[32, 32, 32],
                               forward_sizes=[3, 3, 3],
                               forward_strides=[1, 1, 1],
                               results_path='./results',
                               run_test=False)

    # Training
    vel_predictor.train()
    # Test and Evaluation
    vel_predictor.evaluation()
    # save the trained model
    vel_predictor.save_trained_model()


