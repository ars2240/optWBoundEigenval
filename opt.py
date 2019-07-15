# opt.py
#
# Author: Adam Sandler
# Date: 5/17/19
#
# Contains methods & classes for optimizing NNs with spectral radius regulation
#
#
# Dependencies:
#   Packages: requests, numpy, scipy, sklearn, torch


import gzip
import random
import requests
import math
import numpy as np
import os
import shutil
import sys
from scipy.stats import skewnorm
from scipy.stats import norm
from sklearn.metrics import f1_score
import time
import torch
import torch.utils.data as utils_data


class HVPOperator(object):
    """
    Modified from: https://github.com/noahgolmant/pytorch-hessian-eigenthings

    Use PyTorch autograd for Hessian Vec product calculation
    """

    def __init__(self, model, data, criterion, use_gpu=True, mem_track=False):
        self.model = model  # NN model
        if use_gpu:
            self.model = self.model.cuda()
        self.data = data  # data (inputs, target)
        self.criterion = criterion  # loss function
        self.use_gpu = use_gpu  # whether or not GPUs are used
        self.stored_grad = None  # stored gradient (on CPU)
        self.stored_grad_gpu = None  # stored gradient (on GPU)
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage

    def Hv(self, vec, storedGrad=False):
        # Returns H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        if self.use_gpu:
            vec = vec.cuda()

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            if self.use_gpu:
                grad_vec = self.stored_grad_gpu
            else:
                grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad().double()
            if self.use_gpu:
                self.stored_grad = grad_vec.cpu()
                self.stored_grad_gpu = grad_vec
            else:
                self.stored_grad = grad_vec

        # compute the product
        grad_product = torch.sum(grad_vec * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        if self.use_gpu:
            hessian_vec_prod = hessian_vec_prod.cpu()

        return hessian_vec_prod.data

    def vGHv(self, vec, storedGrad=False):
        # Returns vec*grad H*vec where H is the hessian of the loss w.r.t. the vectorized model parameters

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)
        if self.use_gpu:
            vec = vec.cuda()

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            if self.use_gpu:
                grad_vec = self.stored_grad_gpu
            else:
                grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad().double()
            if self.use_gpu:
                self.stored_grad = grad_vec.cpu()
                self.stored_grad_gpu = grad_vec
            else:
                self.stored_grad = grad_vec

        # compute the product
        grad_product = torch.sum(grad_vec * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True, create_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()
        # compute the product
        grad_product = torch.sum(hessian_vec_prod * vec)

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        vec_grad_hessian_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()

        # check memory usage
        if self.mem_track and self.use_gpu:
            self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])

        if self.use_gpu:
            vec_grad_hessian_vec = vec_grad_hessian_vec.cpu()
        return vec_grad_hessian_vec.data

    def zero_grad(self):
        # Zeros out the gradient info for each parameter in the model
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        # Compute gradient w.r.t loss over all parameters and vectorize
        inputs, target = self.data

        if self.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        output = self.model(inputs)
        if self.criterion.__class__.__name__ == 'KLDivLoss':
            target_onehot = torch.zeros(np.shape(output))
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            loss = self.criterion(output.float(), target_onehot.float())
        else:
            loss = self.criterion(output, target)
        grad_dict = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_dict]))
        grad_vec = grad_vec.double()  # / len(target)  # convert to double if float
        return grad_vec


# Download and parse the dataset
def download(url):
    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    filename = root + '/' + url.split("/")[-1]
    exists = os.path.isfile(filename)
    if not exists:
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    fname = filename[:-3] + '.csv'
    exists = os.path.isfile(fname)
    if not exists:
        with gzip.open(filename, 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return fname


class OptWBoundEignVal(object):
    def __init__(self, model, loss, optimizer, scheduler=None, mu=0, K=0, eps=1e-3, pow_iter_eps=1e-3,
                 use_gpu=False, batch_size=128, min_iter=10, max_iter=100, max_pow_iter=1000, pow_iter=True,
                 max_samples=512, ignore_bad_vals=True, verbose=False, mem_track=False, header=''):
        self.ndim = sum(p.numel() for p in model.parameters())  # number of dimensions
        self.x = 1.0/np.sqrt(self.ndim)*np.ones(self.ndim)  # initial point
        self.f = 0  # loss function value
        self.gradf = np.zeros(self.ndim)  # gradient of f
        self.rho = 0  # spectral radius (maximal absolute value eignevalue)
        self.v = torch.from_numpy(1.0/np.sqrt(self.ndim)*np.ones(self.ndim))  # eigenvector
        self.gradrho = np.zeros(self.ndim)  # gradient of rho
        self.g = 0  # regularizer function
        self.gradg = np.zeros(self.ndim)  # gradient of g
        self.h = 0  # objective function f+mu*g
        self.mu = mu  # coefficient in front of regularizer
        self.K = float(K)  # constant, spectral radius < K
        self.batch_size = batch_size  # batch size
        self.eps = eps  # convergence
        self.pow_iter_eps = pow_iter_eps  # convergence
        self.model = model  # model (from torch)
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = None  # dataloader for training set (from torch)
        self.loss = loss  # loss function (from torch)
        self.optimizer = optimizer  # optimizer function, optional (from torch)
        self.scheduler = scheduler  # learning rate scheduler, optional (from torch)
        self.use_gpu = use_gpu  # whether or not cuda GPUs used
        self.min_iter = min_iter  # minimum number of iterations
        self.max_iter = max_iter  # maximum number of iterations
        self.max_pow_iter = max_pow_iter  # maximum number of power iterations
        self.max_samples = max_samples  # maximum batch size
        self.pow_iter = pow_iter  # whether or not it performs power iteration
        self.hvp_op = None  # Hessian-vector operation object
        self.i = 0  # iteration count
        self.norm = 0  # norm of H*v-lambda*v
        self.val_acc = 0  # validation accuracy (only used if validation set is provided)
        self.best_val_acc = 0  # best validation accuracy (only used if validation set is provided)
        self.best_rho = 0  # spectral radius at best validation accuracy
        self.verbose = verbose  # more extensive read-out
        self.x = None  # input data
        self.y = None  # output data
        name = self.optimizer.__class__.__name__
        if callable(mu):
            mname = 'Func'
        else:
            mname = str(mu)
        # log files
        self.header = header  # header to files
        self.header2 = header + "_" + name + "_mu" + mname + "_K" + str(K)
        self.log_file = "./logs/" + self.header2 + ".log"
        self.verbose_log_file = "./logs/" + self.header2 + "_verbose.log"
        self.ignore_bad_vals = ignore_bad_vals  # whether or not to ignore bad power iteration values
        self.mem_track = mem_track  # whether or not maximum memory usage is tracked
        self.mem_max = 0  # running maximum memory usage

    def comp_rho(self):
        # computes rho, v

        v = self.v  # initial guess for eigenvector (prior eigenvector)

        # initialize lambda and the norm
        lam = 0
        norm = 0

        # power iteration
        for i in range(0, np.min([self.ndim, self.max_pow_iter])):
            vnew = self.hvp_op.Hv(v, storedGrad=True)  # compute H*v

            # if converged, break
            lam = np.dot(vnew, v)  # update eigenvalue
            norm = np.linalg.norm(vnew-lam*v)  # norm of H*v-lambda*v

            if norm < self.pow_iter_eps:
                break

            v = 1.0/np.linalg.norm(vnew)*vnew.double()  # update vector and normalize

        self.v = v  # update eigenvector
        self.rho = np.abs(lam)  # update spectral radius
        self.norm = norm  # update norm

        if norm > self.pow_iter_eps:
            print('Warning: power iteration has not fully converged')
            if self.ignore_bad_vals:
                print('Ignoring rho.')
                self.rho = -1  # if value is discarded due to poor convergence, set to -1
                # as negative values of rho work in the other algorithms and are nonsensical

        if lam == 0:
            print('Warning: rho = 0')

    def comp_gradrho(self):
        # computes grad rho

        self.gradrho = self.hvp_op.vGHv(self.v, storedGrad=True)  # compute v*gradH*v

    def comp_f(self, inputs, target):
        # computes f

        # if using gpu, move data to it
        if self.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()
        output = self.model(inputs)  # compute prediction

        # compute loss
        if self.loss.__class__.__name__ == 'KLDivLoss':
            target_onehot = torch.zeros(np.shape(output))
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            f = self.loss(output.float(), target_onehot.float()).item()
        else:
            f = self.loss(output, target).item()
        return f, output

    def comp_g(self):
        # computes g

        self.comp_rho()
        self.g = np.max([0.0, self.rho - self.K])

    def iter(self):
        # performs one gradient descent iteration

        # if verbose, make header for file
        if self.verbose:
            old_stdout = sys.stdout  # save old output
            if self.i == 0:
                log_file = open(self.verbose_log_file, "w")  # open log file
            else:
                log_file = open(self.verbose_log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('batch\t rho\t norm\t gradf\t gradg')
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

        # compute mu
        if callable(self.mu):
            mu = self.mu(self.i)
        else:
            mu = self.mu

        # pick random batch for estimation of spectral radius at end of epoch
        rbatch = random.randint(0, len(self.dataloader)-1)

        for j, data in enumerate(self.dataloader):

            # store random batch
            if j == rbatch:
                rdata = data

            if self.pow_iter:
                # initialize hessian vector operation class
                self.hvp_op = HVPOperator(self.model, data, self.loss, use_gpu=self.use_gpu)

                self.comp_g()  # compute g

                self.optimizer.zero_grad()  # zero gradient

                # compute grad f
                if self.hvp_op.stored_grad is not None:
                    self.gradf = self.hvp_op.stored_grad.data
                else:
                    self.gradf = torch.zeros(self.ndim).double()  # set gradient to zero

                # compute grad g
                if self.g > 0:
                    self.comp_gradrho()  # compute gradient of rho
                    self.gradg = self.gradrho  # compute g
                else:
                    self.gradg = torch.zeros(self.ndim).double()  # set gradient to zero

                p = self.gradf + mu * self.gradg  # gradient step
                if self.use_gpu:
                    p = p.cuda()  # move to GPU

                i = 0
                for param in self.model.parameters():
                    s = param.data.size()  # number of input & output nodes
                    l = np.product(s)  # total number of parameters
                    param.grad = p[i:(i + l)].view(s).float()  # adjust gradient
                    i += l  # increment
            else:
                # for testing purposes
                self.optimizer.zero_grad()  # zero gradient
                inputs, target = data
                if self.use_gpu:
                    inputs = inputs.cuda()
                    target = target.cuda()
                output = self.model(inputs)
                loss = self.loss(output, target)  # loss function
                loss.backward()  # back prop

            # optimizer step
            self.optimizer.step()

            # if verbose, add values to file
            if self.verbose:
                log_file = open(self.verbose_log_file, "a")  # open log file
                sys.stdout = log_file  # write to log file
                if self.pow_iter:
                    print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm,
                                                    np.linalg.norm(self.gradf.detach().numpy()),
                                                    np.linalg.norm(self.gradg.detach().numpy())))
                else:
                    print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm,
                                                      np.linalg.norm(self.gradf),
                                                      np.linalg.norm(self.gradg)))
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

            # if using GPU, memory cleanup & tracking
            if self.use_gpu:
                torch.cuda.empty_cache()
                # check max memory usage
                if self.mem_track:
                    self.mem_max = np.max([self.mem_max, torch.cuda.memory_allocated()])
                    print('Running Max GPU Memory used (in bytes): %d' % self.mem_max)

        # compute overall estimates
        f_list = []
        size = []
        # compute f on each batch (to avoid memory issues)
        for _, data in enumerate(self.dataloader):
            inputs, target = data
            size.append(len(target))
            f, _ = self.comp_f(inputs, target)
            f_list.append(f)  # compute f on each batch
        self.f = np.average(f_list, weights=size)  # weighted mean of f values
        # initialize hessian vector operation class for random batch
        self.hvp_op = HVPOperator(self.model, rdata, self.loss, use_gpu=self.use_gpu)
        self.comp_g()  # compute g
        self.h = self.f + mu * self.g  # compute objective function

        # adjust learning rate
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self, inputs, target, inputs_valid=None, target_valid=None):

        start = time.time()  # start timer

        self.x = inputs  # input data
        self.y = target  # output data

        # make sure logs folder exists
        if not os.path.exists('./logs'):
            os.mkdir('./logs')

        # make sure models folder exists
        if not os.path.exists('./models'):
            os.mkdir('./models')

        old_stdout = sys.stdout  # save old output

        f_hist = []  # tracks function value after each epoch

        # create dataloader
        train_data = utils_data.TensorDataset(self.x, self.y)
        self.dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size)

        log_file = open(self.log_file, "w")  # open log file
        sys.stdout = log_file  # write to log file

        # header of log file
        if (inputs_valid is None) or (target_valid is None):
            print('epoch\t f\t rho\t h\t norm')
        else:
            print('epoch\t f\t rho\t h\t norm\t val_acc')

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        for self.i in range(0, self.max_iter):
            self.iter()  # take step

            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file

            # add values to log file
            if (inputs_valid is None) or (target_valid is None):
                print('%d\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm))
            else:
                _, self.val_acc = self.test_model(inputs_valid, target_valid)
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
                    self.best_rho = self.rho
                    if self.use_gpu:
                        model = self.model.cpu()
                        torch.save(model.state_dict(), './models/' + self.header2 + '_trained_model_best.pt')
                    else:
                        torch.save(self.model.state_dict(), './models/' + self.header2 + '_trained_model_best.pt')
                print('%d\t %f\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm, self.val_acc))

            # add function value to history log
            f_hist.append(self.h)

            # Save model weights
            if self.use_gpu:
                model = self.model.cpu()
                torch.save(model.state_dict(), './models/' + self.header2 + '_trained_model.pt')
            else:
                torch.save(self.model.state_dict(), './models/' + self.header2 + '_trained_model.pt')

            # check if convergence criteria met
            if self.i >= (self.min_iter - 1):
                coef_var = np.std(f_hist[-10:])/np.abs(np.mean(f_hist[-10:]))
                if coef_var <= self.eps:
                    print(coef_var)
                    break

            if self.i < (self.max_iter - 1):
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

        # compute time elapsed
        end = time.time()
        tTime = end - start
        hrs = math.floor(tTime / 3600)
        tTime = tTime - hrs * 3600
        mins = math.floor(tTime / 60)
        secs = tTime - mins * 60
        print('Time elapsed: %2i hrs, %2i min, %4.2f sec ' % (hrs, mins, secs))

        # best validation accuracy
        print('Best Validation Accuracy:', self.best_val_acc)
        print('Rho:', self.best_rho)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        # compute loss & accuracy on training set
        self.test_train_set(inputs, target)

    def test_model(self, X, y):
        # Computes the loss and accuracy of model on given dataset

        test_data = utils_data.TensorDataset(X, y)
        dataloader = utils_data.DataLoader(test_data, batch_size=self.batch_size)

        f_list = []
        acc_list = []
        f1_list = []
        size = []
        for _, data in enumerate(dataloader):

            inputs, target = data

            # compute loss
            f, ops = self.comp_f(inputs, target)
            f_list.append(f)

            # size of dataset
            size.append(len(target))

            # compute accuracy
            _, predicted = torch.max(ops.data, 1)
            if self.use_gpu:
                target = target.cuda()
            acc = torch.mean((predicted == target).float()).item() * 100
            acc_list.append(acc)

            f1 = f1_score(target, ops)
            f1_list.append(f1)

        test_loss = np.average(f_list, weights=size)  # weighted mean of f values
        test_acc = np.average(acc_list, weights=size)  # weighted mean of accuracy
        test_f1 = np.average(f1_list, weights=size)  # weighted mean of f1 scores

        return test_loss, test_acc, test_f1

    def test_model_best(self, X, y):
        # tests best model, loaded from file

        self.model.load_state_dict(torch.load('./models/' + self.header2 + '_trained_model_best.pt'))

        if self.use_gpu:
            self.model = self.model.cuda()

        return self.test_model(X, y)

    def test_train_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc, f1 = self.test_model_best(X, y)  # test best model

        print('Train Loss:', loss)
        print('Train Accuracy:', acc)
        print('Train F1:', f1)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def test_test_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc, f1 = self.test_model_best(X, y)  # test best model

        print('Test Loss:', loss)
        print('Test Accuracy:', acc)
        print('Test F1:', f1)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def get_prob(self, inputs,  m=[0], sd=[1], skew=[0]):
        # computes log pdf of inputs given mean (m), standard deviation (sd), and skewness (skew)
        if len(m) != 1 or len(sd) != 1 or len(skew) != 1:
            if len(m) == 1:
                if len(sd) > 1:
                    m = np.ones(len(sd))
                else:
                    m = np.ones(len(skew))
            if len(sd) == 1:
                sd = np.ones(len(m))
            if len(skew) == 1:
                skew = np.ones(len(m))

        w = skewnorm.logpdf(inputs, skew, m, sd)
        bad = np.where(np.isinf(w))[0]
        if len(bad) > 0:
            w[bad] = norm.logpdf(inputs[bad, :], m, sd)
        w = np.sum(w, axis=1)

        return w

    def test_model_cov(self, X, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                       train_skew=[0]):
        # Computes the loss and accuracy of model on given dataset

        test_data = utils_data.TensorDataset(X, y)
        dataloader = utils_data.DataLoader(test_data, batch_size=self.batch_size)

        f_list = []
        acc_list = []
        f1_list = []
        size = []
        wm_list = []

        modes = np.logical_or(np.subtract(test_mean, train_mean) != 0, np.subtract(test_sd, train_sd) != 0)
        modes = np.logical_or(modes, np.subtract(test_skew, train_skew) != 0)
        modes = np.where(modes)[0]

        for _, data in enumerate(dataloader):

            inputs, target = data

            feats = inputs.shape[1]
            if len(test_mean) == 1:
                test_mean = test_mean * feats
            if len(test_sd) == 1:
                test_sd = test_sd * feats
            if len(test_skew) == 1:
                test_skew = test_skew * feats
            if len(train_mean) == 1:
                train_mean = train_mean * feats
            if len(train_sd) == 1:
                train_sd = train_sd * feats
            if len(train_skew) == 1:
                train_skew = train_skew * feats

            # compute loss
            f, ops = self.comp_f(inputs, target)
            f_list.append(f)

            # size of dataset
            size.append(len(target))

            # compute accuracy
            _, predicted = torch.max(ops.data, 1)

            w = np.exp(self.get_prob(inputs[:, modes], [test_mean[i] for i in modes], [test_sd[i] for i in modes],
                                     [test_skew[i] for i in modes]) -
                       self.get_prob(inputs[:, modes], [train_mean[i] for i in modes], [train_sd[i] for i in modes],
                                     [train_skew[i] for i in modes]))
            weights = torch.from_numpy(w)
            wm = torch.mean(weights).item()
            wm_list.append(wm)
            weights /= wm * len(target)
            if self.use_gpu:
                target = target.cuda()
                weights = weights.cuda()
            acc = torch.sum(weights.float() * (predicted == target).float()).item() * 100
            acc_list.append(acc)

            f1 = f1_score(target, ops)
            f1_list.append(f1)

        test_loss = np.average(f_list, weights=size)  # weighted mean of f values
        acc_w = np.array(size) * np.array(wm_list)
        acc_w = acc_w/np.sum(acc_w)
        test_acc = np.average(acc_list, weights=acc_w)  # weighted mean of accuracy
        test_f1 = np.average(f1_list, weights=size)  # weighted mean of f1 scores

        return test_loss, test_acc, test_f1

    def test_model_best_cov(self, X, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                       train_skew=[0]):
        # tests best model, loaded from file

        self.model.load_state_dict(torch.load('./models/' + self.header2 + '_trained_model_best.pt'))

        if self.use_gpu:
            self.model = self.model.cuda()

        return self.test_model_cov(X, y, test_mean, test_sd, test_skew, train_mean, train_sd, train_skew)

    def test_cov_shift(self, X, y, test_mean=[0], test_sd=[1], test_skew=[0], train_mean=[0], train_sd=[1],
                       train_skew=[0]):

        # test best model
        loss, acc, f1 = self.test_model_best_cov(X, y, test_mean, test_sd, test_skew, train_mean, train_sd, train_skew)

        print('Test Accuracy:', acc)
        print('Test F1:', f1)


