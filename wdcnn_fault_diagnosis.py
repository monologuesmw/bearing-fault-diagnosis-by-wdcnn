# -*- coding: utf-8 -*-
# @Time    : 2020/1/16 14:24
# @Author  : smw
# @Email   : monologuesmw@163.com
# @File    : wdcnn_fault_diagnosis.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd
import os
import sys
import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def run_wdcnn_fault_diagnosis(phase="TRAIN_AND_TEST", test_size=375, lr=2e-4, batch=40, epoch=100,
                              filename="wdcnn_fault_diagnosis"):
    """
    基于wdcnn模型的Siamese网络进行轴承故障诊断
    :param phase:  由用户选择程序的阶段： TEST_ONLY： 只进行测试【使用本地保存的权重】； TRAIN_AND_TEST : 训练并测试【时间很久】
    :param test_size: 参与测试样本的个数  30条以内速度比较快， 最大为375  【出于对速度的考量】
    :param lr: 学习率  可设置 1e-1, 1e-2, 1e-3, 1e-4, 2e-4, 1e-5
    :param batch: 训练的batch_size 可设置 20, 40, 60  根据服务器内存的大小 可适当增长
    :param epoch: 训练代数   20代比较快， 再大 速度较慢， 但精度取决于训练代数
    :param filename:  结果保存名称
    :return:
    """

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())

    if phase == "TRAIN_AND_TEST":
        checkpoint_dir = ".\\checkpoint\\%s" % TIMESTAMP
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = checkpoint_dir
    elif phase == "TEST_ONLY":
        TIMESTAMP_LOCAL = "LOCAL_MODULES"
        checkpoint_dir = ".\\checkpoint\\%s" % TIMESTAMP_LOCAL
        save_path = checkpoint_dir
    else:
        raise ValueError

    checkpoint_dir = os.path.join(checkpoint_dir, filename) + ".ckpt"
    # log 暂时不要
    # result path
    result_path = ".\\result\\%s"%(TIMESTAMP)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    wdcnn_obj = Wdcnn_Fault_diagnosis(test_size=test_size, lr=lr, batch_size=batch, epoch=epoch,
                                      save_path=save_path, checkpoint_dir=checkpoint_dir)
    # data prepare
    y_test = wdcnn_obj.data_prepare()
    # training stage
    if phase == "TRAIN_AND_TEST":
        wdcnn_obj.fit()
    # test stage
    y_predicts, test_acc = wdcnn_obj.predict()

    # display and results save
    y_predicts = y_predicts.reshape(-1,1)
    y_test = y_test.reshape(-1, 1)
    y_test = y_test[0:test_size]


    cm = confusion_matrix(y_test, y_predicts)
    labels_name = list(range(len(np.unique(y_test))))
    # plot confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title("HAR Confusion Matrix")  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_path, "HAR_cm.png"), format="png")
    plt.show()

    # save result in order to have a look
    acc = np.equal(y_predicts, y_test)
    acc_ = np.mean(acc)
    acc_ = format(acc_, '.4f')
    acc_ = np.array([acc_])
    # y_test = y_test.reshape(-1, 1)
    res = np.hstack((y_predicts, y_test))
    res_Dataframe = pd.DataFrame(res, columns=["predict", "y_label"])
    res_Dataframe.to_csv(os.path.join(result_path, filename + ".csv"))

    np.savetxt(os.path.join(result_path, "accuracy.txt"), acc_, fmt="%s")

    print("finished!")


class Wdcnn_Fault_diagnosis(object):
    def __init__(self, test_size, lr, batch_size, epoch, save_path, checkpoint_dir):
        self.test_size = test_size
        self.lr = lr
        self.batch = batch_size
        self.epoch = epoch
        self.save_path = save_path
        self.checkpoint_dir = checkpoint_dir
        self.num_class = 5
        self.keep_prob = 0.5

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2048, 2], name="inputs")

        self.outputs_label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="output")
        self.outputs_label_raw = tf.reshape(self.outputs_label, shape=[-1, 1])
        self.outputs_label_onehot = tf.placeholder(dtype=tf.float32, shape=[None, self.num_class], name="outputs")

        with tf.variable_scope("WDCNN_Network_Structure") as scope:
            self.train_digits = self.wdcnn_network_structure(is_trained=True)
            scope.reuse_variables()
            self.test_digits = self.wdcnn_network_structure(is_trained=False)

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.outputs_label_onehot,
                                                    logits=self.train_digits,
                                                    scope="loss")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # train acc
        self.train_result = tf.math.argmax(self.train_digits, 1)
        self.train_correlation = tf.equal(self.train_result, tf.math.argmax(self.outputs_label_onehot, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.train_correlation, "float"))

        # test acc
        self.predicts = tf.math.argmax(self.test_digits, 1)
        self.test_correction = tf.equal(self.predicts, tf.math.argmax(self.outputs_label_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.test_correction, "float"))

        # result and accuracy of train
        self.train_result = tf.math.argmax(self.train_digits, 1)
        self.train_correlation = tf.equal(self.train_result, tf.math.argmax(self.outputs_label_onehot, 1))
        self.train_accuracy = tf.reduce_mean(tf.cast(self.train_correlation, "float"))


    def wdcnn_network_structure(self, is_trained):
        with slim.arg_scope([slim.conv1d], padding="same", activation_fn=slim.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            net = slim.conv1d(inputs=self.inputs, num_outputs=16, kernel_size=64, stride=16,
                              scope="conv_1")
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_2")
            net = def_max_pool(net)

            net = slim.conv1d(net, num_outputs=32, kernel_size=3, stride=1, scope="conv_3")
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_4")
            net = def_max_pool(net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=2, stride=1, scope="conv_5")
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_6")
            net = def_max_pool(net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=3, stride=1, scope="conv_7")
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_8")
            net = def_max_pool(net)

            net = slim.conv1d(net, num_outputs=64, kernel_size=3, stride=1, padding="VALID", scope="conv_9")
            def_max_pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="VALID", name="max_pool_10")
            net = def_max_pool(net)

            net = slim.flatten(net, scope="flatten_11")

            net = slim.fully_connected(net, num_outputs=100, activation_fn=tf.nn.sigmoid, scope="fully_connected_12")

            net = slim.dropout(net, keep_prob=self.keep_prob, is_training=is_trained, scope="dropout_13")

            digits_onehot = slim.fully_connected(net, num_outputs=self.num_class, activation_fn=tf.nn.softmax,
                                                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                 weights_regularizer=slim.l2_regularizer(0.005),
                                                 scope="fully_connected_14")
            tf.summary.histogram("fully_connected_14", digits_onehot)
        return digits_onehot

    def data_prepare(self):
        # load data
        data = np.load("Case_Western_Reserve_University_Bearing_fault_data.npz")
        self.x_train, self.x_test, self.y_train, self.y_test, self.n_classes, self.classes = data['arr_0'], data['arr_1'], data['arr_2'], \
                                                               data["arr_3"], data["arr_4"], data["arr_5"]
        return self.y_test

    def fit(self):
        sess = tf.Session()

        saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        print("Currently is the training stage!")
        for step in range(self.epoch):
            avg_cost = 0
            acc = 0
            total_batch = int(self.x_train.shape[0] // self.batch)
            for batch_epoch in range(total_batch):
                batch_x = self.x_train[batch_epoch*self.batch: (batch_epoch+1)*self.batch, :, :]
                batch_y = self.y_train[batch_epoch*self.batch: (batch_epoch+1)*self.batch]
                batch_y_onehot = tf.one_hot(batch_y, depth=self.num_class)
                batch_y_value = sess.run(batch_y_onehot)

                _, cost, acc = sess.run([self.optimizer, self.loss, self.train_accuracy], feed_dict={self.inputs:
                                                                                                batch_x,
                                                                                                self.outputs_label_onehot:
                                                                                                batch_y_value})
                avg_cost += cost/total_batch
                acc += acc/total_batch
            saver.save(sess, save_path=self.checkpoint_dir, global_step=step)
            print("Epoch: %d, Cost: %g, acc: %g"%(step, avg_cost, acc))
        sess.close()

    def predict(self):
        print("Currently is the testing stage!")
        sess = tf.Session()

        saver = tf.train.Saver()
        module_file = tf.train.latest_checkpoint(self.save_path)
        saver.restore(sess, module_file)
        y_test = self.y_test[0: self.test_size]
        x_test = self.x_test[0: self.test_size, :, :]
        y_true_onehot = sess.run(tf.one_hot(y_test, depth=self.num_class))
        y_predicts, test_acc = sess.run([self.predicts, self.accuracy], feed_dict={self.inputs: x_test,
                                                                                      self.outputs_label_onehot:
                                                                                      y_true_onehot})
        return y_predicts, test_acc


if __name__ == '__main__':
    run_wdcnn_fault_diagnosis(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]),
                              int(sys.argv[5]), sys.argv[6])
    # run_wdcnn_fault_diagnosis(phase="TEST_ONLY", test_size=10, lr=2e-4, batch=40, epoch=100,
    #                           filename="wdcnn_fault_diagnosis")



