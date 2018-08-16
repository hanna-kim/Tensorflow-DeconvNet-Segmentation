import os
import random
import tensorflow as tf
# import wget
import tarfile
import numpy as np
import argparse

import time
from datetime import datetime

from utils import input_pipeline
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


class DeconvNet:
    def __init__(self, images, segmentations, use_cpu=False, checkpoint_dir='./checkpoints/'):
        # self.maybe_download_and_extract()

        self.x = images
        self.y = segmentations
        self.build(use_cpu=use_cpu, is_training=True)

        # self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours =1)
        self.saver = tf.train.Saver(tf.global_variables(), \
                                    max_to_keep=5, keep_checkpoint_every_n_hours=1)  # v0.12
        self.checkpoint_dir = checkpoint_dir
        # self.rate=lr
        # start=time.time()

    def maybe_download_and_extract(self):
        """Download and unpack VOC data if data folder only contains the .gitignore file"""
        if os.listdir('data') == ['.gitignore']:
            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG.tar.gz', 'stage_1_train_imgset.tar.gz',
                         'stage_2_train_imgset.tar.gz']
            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data/'

            for filename in filenames:
                wget.download(url + filename, out=os.path.join('data', filename))

                tar = tarfile.open(os.path.join('data', filename))
                tar.extractall(path='data')
                tar.close()

                os.remove(os.path.join('data', filename))

    def restore_session(self):

        global_step = 0
        if not os.path.exists(self.checkpoint_dir):
            raise IOError(self.checkpoint_dir + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if path is None:
                raise IOError('No checkpoint to restore in ' + self.checkpoint_dir)
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)
                global_step = int(path.model_checkpoint_path.split('-')[-1])

        return global_step

        # ! Does not work for now ! But currently I'm working on it -> PR's welcome!

    def predict(self, image):
        self.restore_session()
        return self.prediction.eval(session=self.session, feed_dict={image: [image]})[0]

    # From Github user bcaine, https://github.com/tensorflow/tensorflow/issues/1793
    # @ops.RegisterGradient("MaxPoolWithArgmax")
    def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
        return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                     grad,
                                                     op.outputs[1],
                                                     op.get_attr("ksize"),
                                                     op.get_attr("strides"),
                                                     padding=op.get_attr("padding"))

    def build(self, use_cpu=False, is_training=True):
        '''
        use_cpu allows you to test or train the network even with low GPU memory
        anyway: currently there is no tensorflow CPU support for unpooling respectively
        for the tf.nn.max_pool_with_argmax metod so that GPU support is needed for training
        and prediction
        '''

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:1'

        with tf.device(device):

            # Don't need placeholders when prefetching TFRecords
            # self.x = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='x_data')
            # self.y = tf.placeholder(tf.int64, shape=(None, None, None), name='y_data')

            conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv_1_1')
            bn_1_1 = self.bnorm_relu(conv_1_1, name='bn_1_1', is_training=is_training)
            conv_1_2 = self.conv_layer(bn_1_1, [3, 3, 64, 64], 64, 'conv_1_2')
            bn_1_2 = self.bnorm_relu(conv_1_2, name='bn_1_2', is_training=is_training)

            pool_1, pool_1_argmax = self.pool_layer(bn_1_2)

            conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
            bn_2_1 = self.bnorm_relu(conv_2_1, name='bn_2_1', is_training=is_training)

            conv_2_2 = self.conv_layer(bn_2_1, [3, 3, 128, 128], 128, 'conv_2_2')
            bn_2_2 = self.bnorm_relu(conv_2_2, name='bn_2_2', is_training=is_training)

            pool_2, pool_2_argmax = self.pool_layer(bn_2_2)

            conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
            bn_3_1 = self.bnorm_relu(conv_3_1, name='bn_3_1', is_training=is_training)
            conv_3_2 = self.conv_layer(bn_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
            bn_3_2 = self.bnorm_relu(conv_3_2, name='bn_3_2', is_training=is_training)
            conv_3_3 = self.conv_layer(bn_3_2, [3, 3, 256, 256], 256, 'conv_3_3')
            bn_3_3 = self.bnorm_relu(conv_3_3, name='bn_3_3', is_training=is_training)

            pool_3, pool_3_argmax = self.pool_layer(bn_3_3)

            conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
            bn_4_1 = self.bnorm_relu(conv_4_1, name='bn_4_1', is_training=is_training)
            conv_4_2 = self.conv_layer(bn_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
            bn_4_2 = self.bnorm_relu(conv_4_2, name='bn_4_2', is_training=is_training)
            conv_4_3 = self.conv_layer(bn_4_2, [3, 3, 512, 512], 512, 'conv_4_3')
            bn_4_3 = self.bnorm_relu(conv_4_3, name='bn_4_3', is_training=is_training)

            pool_4, pool_4_argmax = self.pool_layer(bn_4_3)

            conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
            bn_5_1 = self.bnorm_relu(conv_5_1, name='bn_5_1', is_training=is_training)
            conv_5_2 = self.conv_layer(bn_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
            bn_5_2 = self.bnorm_relu(conv_5_2, name='bn_5_2', is_training=is_training)
            conv_5_3 = self.conv_layer(bn_5_2, [3, 3, 512, 512], 512, 'conv_5_3')
            bn_5_3 = self.bnorm_relu(conv_5_3, name='bn_5_3', is_training=is_training)

            pool_5, pool_5_argmax = self.pool_layer(bn_5_3)

            fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
            fbn_6 = self.bnorm_relu(fc_6, name='fbn_6', is_training=is_training)
            fc_7 = self.conv_layer(fbn_6, [1, 1, 4096, 4096], 4096, 'fc_7')
            fbn_7 = self.bnorm_relu(fc_7, name='fbn_7', is_training=is_training)

            deconv_fc_6 = self.deconv_layer(fbn_7, [7, 7, 512, 4096], 512, 'fc6_deconv')
            dfbn_6 = self.bnorm_relu(deconv_fc_6, name='dfbn_6', is_training=is_training)

            # unpool_5 = self.unpool_layer2x2_batch(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))
            unpool_5 = self.unpool_layer2x2_batch(dfbn_6, pool_5_argmax)

            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
            dbn_5_3 = self.bnorm_relu(deconv_5_3, name='dbn_5_3', is_training=is_training)
            deconv_5_2 = self.deconv_layer(dbn_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
            dbn_5_2 = self.bnorm_relu(deconv_5_2, name='dbn_5_2', is_training=is_training)
            deconv_5_1 = self.deconv_layer(dbn_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')
            dbn_5_1 = self.bnorm_relu(deconv_5_1, name='dbn_5_1', is_training=is_training)

            # unpool_4 = self.unpool_layer2x2_batch(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))
            unpool_4 = self.unpool_layer2x2_batch(dbn_5_1, pool_4_argmax)

            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
            dbn_4_3 = self.bnorm_relu(deconv_4_3, name='dbn_4_3', is_training=is_training)
            deconv_4_2 = self.deconv_layer(dbn_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
            dbn_4_2 = self.bnorm_relu(deconv_4_2, name='dbn_4_2', is_training=is_training)
            deconv_4_1 = self.deconv_layer(dbn_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')
            dbn_4_1 = self.bnorm_relu(deconv_4_1, name='dbn_4_1', is_training=is_training)

            # unpool_3 = self.unpool_layer2x2_batch(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))
            unpool_3 = self.unpool_layer2x2_batch(dbn_4_1, pool_3_argmax)

            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
            dbn_3_3 = self.bnorm_relu(deconv_3_3, name='dbn_3_3', is_training=is_training)
            deconv_3_2 = self.deconv_layer(dbn_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
            dbn_3_2 = self.bnorm_relu(deconv_3_2, name='dbn_3_2', is_training=is_training)
            deconv_3_1 = self.deconv_layer(dbn_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')
            dbn_3_1 = self.bnorm_relu(deconv_3_1, name='dbn_3_1', is_training=is_training)

            # unpool_2 = self.unpool_layer2x2_batch(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))
            unpool_2 = self.unpool_layer2x2_batch(dbn_3_1, pool_2_argmax)

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
            dbn_2_2 = self.bnorm_relu(deconv_2_2, name='dbn_2_2', is_training=is_training)
            deconv_2_1 = self.deconv_layer(dbn_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')
            dbn_2_1 = self.bnorm_relu(deconv_2_1, name='dbn_4_1', is_training=is_training)

            # unpool_1 = self.unpool_layer2x2_batch(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))
            unpool_1 = self.unpool_layer2x2_batch(dbn_2_1, pool_1_argmax)

            deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
            dbn_1_2 = self.bnorm_relu(deconv_1_2, name='dbn_1_2', is_training=is_training)
            deconv_1_1 = self.deconv_layer(dbn_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')
            dbn_1_1 = self.bnorm_relu(deconv_1_1, name='dbn_1_1', is_training=is_training)

            score_1 = self.deconv_layer(dbn_1_1, [1, 1, 21, 32], 21, 'score_1')
            dbn_score = self.bnorm_relu(score_1, name='dbn_score', is_training=is_training)

            self.logits = tf.reshape(dbn_score, (-1, 21))

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.cast(tf.reshape(self.y, [-1]), tf.int64), logits=self.logits, name='x_entropy')

            self.loss_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

            self.train_step = tf.train.AdamOptimizer(args.lr).minimize(self.loss_mean)

            summary_op = tf.summary.merge_all()  # v0.12
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.cast(tf.reshape(self.y, [-1]), tf.int64), name='x_entropy')
            # loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            # loss = tf.Print(loss, [loss], "loss: ")

            # self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

            # self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), dimension=3)
            # commented out for now, was throwing errors, calculating loss is fine.
            # self.accuracy = tf.reduce_sum(tf.pow(self.prediction - self.y, 2))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding) + b)

    def bnorm_relu(self, x, name, is_training, decay=0.999):

        pop_mean = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=False)
        gamma = tf.Variable(tf.ones([x.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))  # size coordinate will be needed
        if is_training:
            mean_, var_ = tf.nn.moments(x, axes=[0, 1, 2], name=name + '_moment', keep_dims=False)
            train_mean = tf.assign(pop_mean, pop_mean * decay + mean_ * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + var_ * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                bnorm_ = tf.nn.batch_normalization(x, mean=mean_, variance=var_, offset=beta,
                                                   scale=gamma, variance_epsilon=1e-3, name=name + '_bnorm')

        else:
            bnorm_ = tf.nn.batch_normalization(x, mean=pop_mean, variance=pop_var, offset=beta, scale=gamma,
                                               variance_epsilon=1e-3, name=name + '_bnorm_testing')

        relu_ = tf.nn.relu(bnorm_, name=name + '_relu')
        return relu_

    def pool_layer(self, x):
        '''
        see description of build method
        '''
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
        W = self.weight_variable(W_shape)
        b = self.bias_variable([b_shape])

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    # def unpool_layer2x2(self, x, raveled_argmax, out_shape):
    # argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    # output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

    # height = tf.shape(output)[0]
    # width = tf.shape(output)[1]
    # channels = tf.shape(output)[2]

    # t1 = tf.to_int64(tf.range(channels))
    # t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    # t1 = tf.reshape(t1, [-1, channels])
    # t1 = tf.transpose(t1, perm=[1, 0])
    # t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

    # t2 = tf.squeeze(argmax)
    # t2 = tf.stack((t2[0], t2[1]), axis=0)
    # t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

    # t = tf.concat([t2, t1], 3)
    # indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

    # x1 = tf.squeeze(x)
    # x1 = tf.reshape(x1, [-1, channels])
    # x1 = tf.transpose(x1, perm=[1, 0])
    # values = tf.reshape(x1, [-1])

    # delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    # return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def unpool_layer2x2_batch(self, bottom, argmax):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = self.unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat([t2, t3, t1], 4)
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])
        return tf.scatter_nd(indices, values, tf.to_int64(top_shape))

    def train(self, train_stage=1):
        # train step
        ###########this part moved to build() ###########################################
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #   labels=tf.cast(tf.reshape(deconvnet.y, [-1]), tf.int64), logits=self.logits, name='x_entropy')

        # loss_mean = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

        # train_step = tf.train.AdamOptimizer(args.lr).minimize(loss_mean)

        # summary_op = tf.summary.merge_all()  # v0.12
        ######################################################################################

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # summary_writer = tf.train.SummaryWriter(args.train_dir, sess.graph)
        if train_stage == 1:
            self.summary_writer = tf.summary.FileWriter(args.train_dir, sess.graph)  # v0.12
        # training_summary = tf.scalar_summary("loss", loss_mean)
        self.training_summary = tf.summary.scalar("loss", self.loss_mean)  # v0.12

        try:
            if train_stage == 1:
                step = 0
            else:
                step = self.restore_session()
            while not coord.should_stop():
                start_time = time.time()
                _, loss_val, train_sum = sess.run([self.train_step, self.loss_mean, self.training_summary],
                                                  options=run_options)
                elapsed = time.time() - start_time
                self.summary_writer.add_summary(train_sum, step)
                # sess.run(self.saver.save(sess=sess, save_path=self.checkpoint_dir, global_step=step), options=run_options)
                self.saver.save(sess=sess, save_path=self.checkpoint_dir + 'model', global_step=step)
                # print sess.run(deconvnet.prediction)

                assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

                step += 1

                if step % 1 == 0:
                    num_examples_per_step = args.batch_size
                    examples_per_sec = num_examples_per_step / elapsed
                    sec_per_batch = float(elapsed)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_val,
                                        examples_per_sec, sec_per_batch))


        except tf.errors.OutOfRangeError:
            print
            'Done training -- epoch limit reached'
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    # Using argparse over tf.FLAGS as I find they behave better in ipython
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_record1', help="training tfrecord file", default="tfrecords/pascalvoc2012.tfrecords")
    parser.add_argument('--train_record2', help="training tfrecord file",
                        default="tfrecords/pascalvoc2012_stage2.tfrecords")
    parser.add_argument('--train_dir', help="where to log training", default="train_log")
    parser.add_argument('--batch_size', help="batch size", type=int, default=10)
    parser.add_argument('--num_epochs', help="number of epochs.", type=int, default=50)
    parser.add_argument('--lr', help="learning rate", type=float, default=1e-4)
    args = parser.parse_args()

    trn_images_batch, trn_segmentations_batch = input_pipeline(
        args.train_record1,
        args.batch_size,
        args.num_epochs)

    deconvnet = DeconvNet(trn_images_batch, trn_segmentations_batch, use_cpu=False)
    # a network is built as deconvnet is declared
    init_global = tf.global_variables_initializer()  # v0.12
    init_locals = tf.local_variables_initializer()  # v0.12

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        sess.run([init_global, init_locals], options=run_options)
        deconvnet.train()  # stage1 training

        trn_images_batch2, trn_segmentations_batch2 = input_pipeline(
            args.train_record2,
            args.batch_size,
            args.num_epochs)
        deconvnet.x = trn_images_batch2
        deconvnet.y = trn_segmentations_batch2

        # deconvnet.build(use_cpu=False, is_training=True)
        # Do I really need to build again?
        deconvnet.train(train_stage=2)
