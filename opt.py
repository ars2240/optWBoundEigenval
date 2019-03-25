import numpy as np
import os
import sys
import torch
import torch.utils.data as utils_data


class HVPOperator(object):
    """
    Modified from: https://github.com/noahgolmant/pytorch-hessian-eigenthings

    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_samples: max number of examples per batch using all GPUs.
    """

    def __init__(self, model, data, criterion, use_gpu=True, max_iter=100):
        size = int(sum(p.numel() for p in model.parameters()))
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.data = data
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.stored_grad = None
        self.max_iter = max_iter

    def Hv(self, vec, storedGrad=False):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad().double()
            self.stored_grad = grad_vec
        # compute the product
        grad_product = torch.sum(grad_vec * vec)
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), retain_graph=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()
        return hessian_vec_prod

    def vGHv(self, vec, storedGrad=False):
        """
        Returns vec*grad H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """

        # convert numpy array to torch tensor
        if type(vec) is np.ndarray:
            vec = torch.from_numpy(vec)

        vec = vec.double()  # convert to double if float

        # compute original gradient, tracking computation graph
        self.zero_grad()
        if storedGrad and (self.stored_grad is not None):
            grad_vec = self.stored_grad
        else:
            grad_vec = self.prepare_grad()
            self.stored_grad = grad_vec
        # compute the product
        grad_product = torch.sum(grad_vec * vec)
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), create_graph=True, retain_graph=True,
                                        allow_unused=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()
        # compute the product
        grad_product = torch.sum(hessian_vec_prod * vec)
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_product, self.model.parameters(), create_graph=True, retain_graph=True,
                                        allow_unused=True)
        # concatenate the results over the different components of the network
        vec_grad_hessian_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_grad])).double()
        return vec_grad_hessian_vec

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        inputs, target = self.data

        if self.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()

        output = self.model(inputs)
        try:
            loss = self.criterion(output, target)
        except RuntimeError:
            try:
                target_onehot = torch.zeros(np.shape(output))
                target_onehot.scatter_(1, target.view(-1, 1), 1)
                loss = self.criterion(output.float(), target_onehot.float())
            except RuntimeError:
                print(np.shape(target))
                print(target)
                print(np.shape(target_onehot))
        grad_dict = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat(tuple([g.contiguous().view(-1) for g in grad_dict]))
        self.grad_vec = grad_vec.double()  # convert to double if float
        return self.grad_vec


class OptWBoundEignVal(object):
    def __init__(self, model, loss, optimizer, scheduler=None, mu=0, K=0, eps=1e-3, pow_iter_eps=1e-3,
                 use_gpu=False, batch_size=128, min_iter=10, max_iter=100, max_pow_iter=1000, max_samples=512,
                 ignore_bad_vals=True, verbose=False):
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
        self.mu = float(mu)  # coefficient in front of regularizer
        self.K = float(K)  # constant, spectral radius < K
        self.batch_size = batch_size  # batch size
        self.eps = eps  # convergence
        self.pow_iter_eps = pow_iter_eps  # convergence
        self.model = model  # model (from torch)
        self.dataloader = None  # dataloader for training set (from torch)
        self.loss = loss  # loss function (from torch)
        self.optimizer = optimizer  # optimizer function, optional (from torch)
        self.scheduler = scheduler  # learning rate scheduler, optional (from torch)
        self.use_gpu = use_gpu  # whether or not cuda GPUs used
        self.min_iter = min_iter  # minimum number of iterations
        self.max_iter = max_iter  # maximum number of iterations
        self.max_pow_iter = max_pow_iter  # maximum number of power iterations
        self.max_samples = max_samples  # maximum batch size
        self.hvp_op = None  # Hessian-vector operation object
        self.i = 0  # iteration count
        self.norm = 0  # norm of H*v-lambda*v
        self.val_acc = 0  # validation accuracy (only used if validation set is provided)
        self.best_val_acc = 0  # best validation accuracy (only used if validation set is provided)
        self.verbose = verbose  # more extensive read-out
        self.log_file = "./logs/MNIST_Adam_mu" + str(mu) + "_K" + str(K) + ".log"  # log file
        self.verbose_log_file = "./logs/MNIST_Adam_mu" + str(mu) + "_K" + str(K) + "_verbose.log"  # log file
        self.ignore_bad_vals = ignore_bad_vals  # whether or not to ignore bad power iteration values

    def comp_rho(self):
        # computes rho, v
        v = self.v  # initial guess for eigenvector (prior eigenvector)

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
        self.rho = np.abs(np.dot(vnew, v))  # update eigenvalue
        self.norm = norm  # update norm

        if norm > self.pow_iter_eps:
            print('Warning: power iteration has not fully converged')
            if self.ignore_bad_vals:
                self.rho = -1  # if value is discarded due to poor convergence, set to -1
                # as negative values of rho work in the other algorithms and are nonsensical

    def comp_gradrho(self):
        # computes grad rho
        self.gradrho = self.hvp_op.vGHv(self.v, storedGrad=True)  # compute v*gradH*v

    def comp_f(self, inputs, target):
        # computes f
        output = self.model(inputs)

        try:
            self.f = self.loss(output, target).item()
        except RuntimeError:
            target_onehot = torch.zeros(np.shape(output))
            target_onehot.scatter_(1, target.view(-1, 1), 1)
            self.f = self.loss(output.float(), target_onehot.float()).item()

    def comp_g(self):
        # computes g
        self.comp_rho()
        self.g = np.max([0.0, self.rho - self.K])

    def iter(self):
        # performs one gradient descent iteration

        # adjust learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        if self.verbose:
            old_stdout = sys.stdout  # save old output
            if self.i == 0:
                log_file = open(self.verbose_log_file, "w")  # open log file
            else:
                log_file = open(self.verbose_log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file
            print('batch\t rho\t norm\t gradf\t gradg')

        for j, data in enumerate(self.dataloader):

            self.hvp_op = HVPOperator(self.model, data, self.loss, use_gpu=self.use_gpu)

            self.comp_g()  # compute g

            # compute grad f
            if self.hvp_op.stored_grad is not None:
                self.gradf = self.hvp_op.stored_grad
            else:
                self.gradf = torch.zeros(self.ndim).double()  # set gradient to zero

            # compute grad g
            if self.g > 0:
                self.comp_gradrho()  # compute gradient of rho
                self.gradg = self.gradrho  # compute g
            else:
                self.gradg = torch.zeros(self.ndim).double()  # set gradient to zero

            # compute mu
            if callable(self.mu):
                mu = self.mu(self.i)
            else:
                mu = self.mu

            p = self.gradf + mu * self.gradg  # gradient step
            i = 0
            for param in self.model.parameters():
                s = param.data.size()
                l = np.product(s)
                param.grad = p[i:(i + l)].view(s).float()  # adjust gradient
                i += l

            # optimizer step
            self.optimizer.step()

            if self.verbose:
                print('%d\t %f\t %f\t %f\t %f' % (j, self.rho, self.norm, np.linalg.norm(self.gradf.detach().numpy()),
                                                  np.linalg.norm(self.gradg.detach().numpy())))

        # compute overall estimates
        inputs, target = data
        self.comp_f(inputs, target)  # compute f
        self.comp_g()  # compute g
        self.h = self.f + self.mu * self.g  # compute objective function

        if self.verbose:
            log_file.close()  # close log file
            sys.stdout = old_stdout  # reset output

    def train(self, inputs, target, inputs_valid=None, target_valid=None):

        # make sure logs file exists
        if not os.path.exists('./logs'):
            os.mkdir('./logs')

        old_stdout = sys.stdout  # save old output

        f_hist = []
        train_data = utils_data.TensorDataset(inputs, target)
        self.dataloader = utils_data.DataLoader(train_data, batch_size=self.batch_size)

        log_file = open(self.log_file, "w")  # open log file
        sys.stdout = log_file  # write to log file

        if (inputs_valid is None) or (target_valid is None):
            print('epoch\t f\t rho\t h\t norm')
        else:
            print('epoch\t f\t rho\t h\t norm\t val_acc')

        log_file.close()  # close log file

        for self.i in range(0, self.max_iter):
            self.iter()  # take step

            log_file = open(self.log_file, "a")  # open log file
            sys.stdout = log_file  # write to log file

            if (inputs_valid is None) or (target_valid is None):
                print('%d\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm))
            else:
                _, self.val_acc = self.test_model(inputs_valid, target_valid)
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
                    torch.save(self.model.state_dict(), 'trained_model_best.pt')
                print('%d\t %f\t %f\t %f\t %f\t %f' % (self.i, self.f, self.rho, self.h, self.norm, self.val_acc))

            f_hist.append(self.h)
            if self.i >= (self.min_iter-1):
                coef_var = np.std(f_hist[-10:])/np.abs(np.mean(f_hist[-10:]))
                if coef_var <= self.eps:
                    print(coef_var)
                    break
            if self.i < (self.max_iter - 1):
                log_file.close()  # close log file
                sys.stdout = old_stdout  # reset output

        # Save model weights
        torch.save(self.model.state_dict(), 'trained_model.pt')

        print('Best Validation Loss:', self.best_val_acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

        self.test_train_set(inputs, target)

    def test_model(self, X, y):
        """
        Tests the model using stored network weights.
        Please ensure that this code will allow me to test your model on testing data.
        """

        # compute loss and accuracy
        ops = self.model(X)
        _, predicted = torch.max(ops.data, 1)
        try:
            test_loss = self.loss(ops, y).item()
        except RuntimeError:
            target_onehot = torch.zeros(np.shape(ops))
            target_onehot.scatter_(1, y.view(-1, 1), 1)
            test_loss = self.loss(ops.float(), target_onehot.float()).item()
        test_acc = torch.mean((predicted == y).float()).item() * 100

        return test_loss, test_acc

    def test_model_best(self, X, y):

        self.model.load_state_dict(torch.load('trained_model_best.pt'))

        return self.test_model(X, y)

    def test_train_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc = self.test_model_best(X, y)

        print('Train Loss:', loss)
        print('Train Accuracy:', acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output

    def test_test_set(self, X, y):
        old_stdout = sys.stdout  # save old output
        log_file = open(self.log_file, "a")  # open log file
        sys.stdout = log_file  # write to log file

        loss, acc = self.test_model_best(X, y)

        print('Test Loss:', loss)
        print('Test Accuracy:', acc)

        log_file.close()  # close log file
        sys.stdout = old_stdout  # reset output



