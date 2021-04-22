import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array
from ares.loss import CrossEntropyLoss
import scipy.stats as st
import math
import os
import sys
from sys import path
path.append(sys.path[0]+'/attacker')


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session, self.dataset = model, batch_size, session, dataset
        if dataset == 'imagenet':
            self.class_num = 1000
        elif dataset == 'cifar10':
            self.class_num = 10
        # dataset == "imagenet" or "cifar10"
        # loss = CrossEntropyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.other_ys_ph = get_ys_ph(model, batch_size)
        self.logits = self.model.logits(self.xs_ph)
        self.raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys_ph, logits=self.logits)

        label_mask = tf.one_hot(self.ys_ph,
                                self.class_num,
                                on_value=1.0,
                                off_value=0.0,
                                dtype=tf.float32)

        self.softmax = tf.nn.softmax(self.logits)
        correct_logit = tf.reduce_sum(label_mask * self.softmax, axis=1)
        wrong_logit = tf.reduce_max((1 - label_mask) * self.softmax - 1e4 * label_mask, axis=1)  # 用logits，不要纯用softmax
        # self.margin_loss = wrong_logit - correct_logit  # 更换损失函数，这个损失函数很猛，效果比交叉熵好的多了
        self.margin_loss = -tf.nn.relu(correct_logit - wrong_logit + 50.)  # 这个是CWloss

        cifar10_correct_logit = tf.reduce_sum(label_mask * self.logits, axis=1)
        cifar10_wrong_logit = tf.reduce_max((1 - label_mask) * self.logits - 1e4 * label_mask,
                                            axis=1)  # 用logits，不要纯用softmax
        self.cifar10_margin_loss = cifar10_wrong_logit - cifar10_correct_logit  # 更换损失函数，这个损失函数很猛，效果比交叉熵好的多了

        self.raw_grad = tf.gradients(self.raw_loss, self.xs_ph)[0]
        self.grad = tf.gradients(self.margin_loss, self.xs_ph)[0]
        self.cifar10_grad = tf.gradients(self.cifar10_margin_loss, self.xs_ph)[0]

        self.rand_direct = tf.Variable(np.zeros((self.batch_size, self.class_num)).astype(np.float32),
                                       name='rand_direct')
        self.input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.class_num])
        self.assign_op = self.rand_direct.assign(self.input_placeholder)

        self.ODI_loss = tf.tensordot(self.softmax, self.rand_direct, axes=[[0, 1], [0, 1]])  # 用logits，不要纯用softmax
        self.grad_ODI = tf.gradients(self.ODI_loss, self.xs_ph)[0]

        self.cifar10_ODI_loss = tf.tensordot(self.logits, self.rand_direct,
                                             axes=[[0, 1], [0, 1]])  # 用logits，不要纯用softmax
        self.cifar10_grad_ODI = tf.gradients(self.cifar10_ODI_loss, self.xs_ph)[0]

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-8
            self.set_eps = self.eps
            print("——————————————————————————————————————————————————————————————————————————————————————")
            print(self.set_eps)
            self.alpha = self.eps
            self.ODI_alpha = self.eps

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        # min_step = self.alpha / self.iteration
        update_vector = np.ones(xs.shape)  # 攻击成功则停止更新
        replace_vector = np.zeros(xs.shape)
        xs_adv = xs

        important_logits_all = self._session.run(self.logits,
                                                 feed_dict={self.xs_ph: xs,
                                                            self.ys_ph: ys})  # 利用logits判断是否攻击成功，失败的可以再重启
        # print(np.max(important_logits_all))
        important_logits = np.max(important_logits_all)
        important_logits_min = np.min(important_logits_all)

        if self.dataset == 'cifar10':
            if important_logits < 3.2 :
                print("MMC和feature_scatter定制")
                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False
                for i in range(start_times):
                    # best_per = xs #设置冷启动，热启动对imagenet基本无效

                    if rand:  # 这个扰动很垃圾，没什么用
                        x = best_per + np.random.uniform(-self.eps, self.eps, xs.shape)  # 这里设置热启动还是冷启动
                        x = np.clip(x, xs_lo, xs_hi)
                        best_per = np.clip(x, self.model.x_min, self.model.x_max)  # ensure valid pixel range

                    rand_vector = np.random.uniform(-1.0, 1.0, (self.batch_size, self.class_num))
                    self._session.run(self.assign_op,
                                      feed_dict={self.input_placeholder: rand_vector.astype(np.float32)})
                    for k in range(ODI_times):
                        use_alpha = self.ODI_alpha
                        logits, grad = self._session.run([self.logits, self.cifar10_grad_ODI],
                                                         feed_dict={self.xs_ph: best_per,
                                                                    self.ys_ph: ys})  # 使用热启动 ，要设置self.rand = false
                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

                    for j in range(iter_times):
                        iteration = (j - 0) % iter_times
                        use_alpha = min_step + (self.alpha - min_step) * (
                                1 + math.cos(math.pi * iteration / iter_times)) / 2  # 余弦退火调整学习率
                        # use_alpha = use_alpha/4
                        logits, grad = self._session.run([self.logits, self.cifar10_grad],
                                                         feed_dict={self.xs_ph: best_per, self.ys_ph: ys})

                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False
                for i in range(start_times):
                    # best_per = xs #设置冷启动，热启动对imagenet基本无效

                    if rand:  # 这个扰动很垃圾，没什么用
                        x = best_per + np.random.uniform(-self.eps, self.eps, xs.shape)  # 这里设置热启动还是冷启动
                        x = np.clip(x, xs_lo, xs_hi)
                        best_per = np.clip(x, self.model.x_min, self.model.x_max)  # ensure valid pixel range

                    rand_vector = np.random.uniform(-1.0, 1.0, (self.batch_size, self.class_num))
                    self._session.run(self.assign_op,
                                      feed_dict={self.input_placeholder: rand_vector.astype(np.float32)})
                    for k in range(ODI_times):
                        use_alpha = self.ODI_alpha
                        logits, grad = self._session.run([self.logits, self.cifar10_grad_ODI],
                                                         feed_dict={self.xs_ph: best_per,
                                                                    self.ys_ph: ys})  # 使用热启动 ，要设置self.rand = false
                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

                    for j in range(iter_times):
                        iteration = (j - 0) % iter_times
                        use_alpha = min_step + (self.alpha - min_step) * (
                                1 + math.cos(math.pi * iteration / iter_times)) / 2  # 余弦退火调整学习率
                        # use_alpha = use_alpha/4
                        logits, grad = self._session.run([self.logits, self.cifar10_grad],
                                                         feed_dict={self.xs_ph: best_per, self.ys_ph: ys})

                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

            else:
                print("使用softmax")
                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False
                for i in range(start_times):
                    # best_per = xs #设置冷启动，热启动对imagenet基本无效

                    if rand:  # 这个扰动很垃圾，没什么用
                        x = best_per + np.random.uniform(-self.eps, self.eps, xs.shape)  # 这里设置热启动还是冷启动
                        x = np.clip(x, xs_lo, xs_hi)
                        best_per = np.clip(x, self.model.x_min, self.model.x_max)  # ensure valid pixel range

                    rand_vector = np.random.uniform(-1.0, 1.0, (self.batch_size, self.class_num))
                    self._session.run(self.assign_op,
                                      feed_dict={self.input_placeholder: rand_vector.astype(np.float32)})
                    for k in range(ODI_times):
                        use_alpha = self.ODI_alpha
                        logits, grad = self._session.run([self.logits, self.grad_ODI],
                                                         feed_dict={self.xs_ph: best_per,
                                                                    self.ys_ph: ys})  # 使用热启动 ，要设置self.rand = false

                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

                    for j in range(iter_times):
                        iteration = (j - 0) % iter_times
                        use_alpha = min_step + (self.alpha - min_step) * (
                                1 + math.cos(math.pi * iteration / iter_times)) / 2  # 余弦退火调整学习率
                        # use_alpha = use_alpha/4
                        logits, grad = self._session.run([self.logits, self.grad],
                                                         feed_dict={self.xs_ph: best_per, self.ys_ph: ys})

                        predict = np.argmax(logits, axis=1)

                        flag = (predict != ys)
                        xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                        update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新

                        grad_sign = np.sign(grad)

                        xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                        xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                        best_per = xxxx

        start_times = 5
        ODI_times = 2
        iter_times = 18
        min_step = self.alpha / 4
        best_per = xs_adv
        rand = False
        if self.dataset == 'imagenet':
            for i in range(start_times):
                # best_per = xs #设置冷启动，热启动对imagenet基本无效

                if rand:  # 这个扰动很垃圾，没什么用
                    x = best_per + np.random.uniform(-self.eps, self.eps, xs.shape)  # 这里设置热启动还是冷启动
                    x = np.clip(x, xs_lo, xs_hi)
                    best_per = np.clip(x, self.model.x_min, self.model.x_max)  # ensure valid pixel range

                rand_vector = np.random.uniform(-1.0, 1.0, (self.batch_size, self.class_num))
                self._session.run(self.assign_op, feed_dict={self.input_placeholder: rand_vector.astype(np.float32)})
                for k in range(ODI_times):
                    use_alpha = self.ODI_alpha
                    logits, grad = self._session.run([self.logits, self.grad_ODI], feed_dict={self.xs_ph: best_per,
                                                                                              self.ys_ph: ys})  # 使用热启动 ，要设置self.rand = false
                    predict = np.argmax(logits, axis=1)

                    flag = (predict != ys)
                    xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                    update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新
                    grad_sign = np.sign(grad)

                    xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                    xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                    best_per = xxxx

                for j in range(iter_times):
                    iteration = (j - 0) % iter_times
                    use_alpha = min_step + (self.alpha - min_step) * (
                            1 + math.cos(math.pi * iteration / iter_times)) / 2  # 余弦退火调整学习率
                    # use_alpha = use_alpha/4
                    logits, grad = self._session.run([self.logits, self.grad],
                                                     feed_dict={self.xs_ph: best_per, self.ys_ph: ys})

                    predict = np.argmax(logits, axis=1)

                    flag = (predict != ys)
                    xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                    update_vector[flag, :] = replace_vector[flag, :]  # 攻击成功即停止更新
                    grad_sign = np.sign(grad)

                    xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                    xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                    best_per = xxxx


        return xs_adv
