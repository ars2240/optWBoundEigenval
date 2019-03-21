import numpy as np
import scipy.io


class Sigmoid(object):
    def __init__(self, x):
        self.x = x

    def f(self):
        return 1.0 / (1.0 + np.exp(-self.x))

    def df(self):
        f = self.f()
        return np.multiply(f, 1.0-f)

    def d2f(self):
        f = self.f()
        return np.multiply(np.multiply(f, 1.0-f), 1.0-2.0*f)

    def d3f(self):
        f = self.f()
        return np.multiply(np.multiply(f, 1.0-f), 6.0*np.power(f, 2)-6.0*f+1.0)


class MeanSquaredError(object):
    def __init__(self, yhat, y):
        self.y = y
        self.yhat = yhat

    def f(self):
        return 0.5*np.sum(np.power(self.yhat-self.y, 2))

    def df(self):
        return self.yhat-self.y

    def d2f(self):
        return 1.0

    def d3f(self):
        return 0.0


class ROp(object):
    def __init__(self, fname):
        self.mdict = scipy.io.loadmat(fname)  # import dataset from matlab
        self.x = self.mdict.get('x')
        self.y = self.mdict.get('y')
        self.w = self.mdict.get('w')
        self.b = self.mdict.get('b')
        self.v = self.mdict.get('v')
        self.n = len(self.x)
        self.len = len(self.w)
        self.xhat = {}
        self.yhat = {}
        self.dEdy = {}
        self.dEdx = {}
        self.dEdw = {}
        self.rx = {}
        self.ry = {}
        self.rdEdy = {}
        self.rdEdx = {}
        self.rdEdw = {}
        self.r2x = {}
        self.r2y = {}
        self.r2dEdy = {}
        self.r2dEdx = {}
        self.r2dEdw = {}

    def forward(self):
        for i in range(0, self.len):

            # compute x
            if i == 0:
                self.xhat[i] = np.dot(self.w[(i, 0)], self.x)+self.b[(i, 0)]
            else:
                self.xhat[i] = np.dot(self.w[(i, 0)], self.yhat[i-1])+self.b[(i, 0)]

            # compute y
            sig = Sigmoid(self.xhat[i])
            self.yhat[i] = sig.f()

            # compute R{x}
            v = np.reshape(self.v[(i * self.n ** 2):((i + 1) * self.n ** 2)], (self.n, self.n), order='F')
            if i == 0:
                self.rx[i] = np.dot(v, self.x)
            else:
                self.rx[i] = np.dot(v, self.yhat[i-1])
                self.rx[i] += np.dot(self.w[(i, 0)], self.ry[i-1])

            # compute R{y}
            self.ry[i] = np.multiply(self.rx[i], sig.df())

            # compute R^2{x}
            if i == 0:
                self.r2x[i] = 0.0
            else:
                self.r2x[i] = np.dot(self.w[(i, 0)], self.r2y[i-1])
                self.r2x[i] += 2.0*np.dot(v, self.ry[i-1])

            # compute R^2{y}
            self.r2y[i] = np.multiply(self.r2x[i], sig.df()) + np.multiply(np.power(self.rx[i], 2), sig.d2f())

    def backward(self):
        for i in range(self.len-1, -1, -1):

            # check if forward pass computed
            if not self.yhat:
                raise Exception('Run forward pass before backward.')

            # compute dE/dy
            if i == (self.len-1):
                err = MeanSquaredError(self.yhat[i], self.y)
                self.dEdy[i] = err.df()
            else:
                self.dEdy[i] = np.transpose(np.dot(np.transpose(self.dEdx[i+1]), self.w[(i+1, 0)]))

            # compute dE/dx
            sig = Sigmoid(self.xhat[i])
            self.dEdx[i] = np.multiply(sig.df(), self.dEdy[i])

            # compute dE/dw
            if i == 0:
                self.dEdw[i] = np.outer(self.dEdx[i], self.x)
            else:
                self.dEdw[i] = np.outer(self.dEdx[i], self.yhat[i - 1])

            # compute R{dE/dy}
            if i == (self.len - 1):
                self.rdEdy[i] = np.multiply(err.d2f(), self.ry[i])
            else:
                self.rdEdy[i] = np.transpose(np.dot(np.transpose(self.rdEdx[i+1]), self.w[(i+1, 0)]))
                v = np.reshape(self.v[((i + 1) * self.n ** 2):((i + 2) * self.n ** 2)], (self.n, self.n), order='F')
                self.rdEdy[i] += np.transpose(np.dot(np.transpose(self.dEdx[i+1]), v))

            # compute R{dE/dx}
            self.rdEdx[i] = np.multiply(sig.df(), self.rdEdy[i])
            self.rdEdx[i] += np.multiply(np.multiply(self.rx[i], sig.d2f()), self.dEdy[i])

            # compute R{dE/dw}
            if i == 0:
                self.rdEdw[i] = np.outer(self.rdEdx[i], self.x)
            else:
                self.rdEdw[i] = np.outer(self.rdEdx[i], self.yhat[i - 1]) + np.outer(self.dEdx[i], self.ry[i - 1])

            # compute R^2{dE/dy}
            if i == (self.len - 1):
                self.r2dEdy[i] = np.multiply(err.d3f(), np.power(self.ry[i], 2)) + np.multiply(err.d2f(), self.r2y[i])
            else:
                self.r2dEdy[i] = np.transpose(np.dot(np.transpose(self.r2dEdx[i+1]), self.w[(i+1, 0)]))
                self.r2dEdy[i] += 2.0*np.transpose(np.dot(np.transpose(self.rdEdx[i+1]), v))

            # compute R^2{dE/dx}
            self.r2dEdx[i] = 2.0 * np.multiply(np.multiply(self.rx[i], sig.d2f()), self.rdEdy[i])
            self.r2dEdx[i] += np.multiply(sig.df(), self.r2dEdy[i])
            self.r2dEdx[i] += np.multiply(np.multiply(self.r2x[i], sig.d2f()), self.dEdy[i])
            self.r2dEdx[i] += np.multiply(np.multiply(np.power(self.rx[i], 2), sig.d3f()), self.dEdy[i])

            # compute R^2{dE/dw}
            if i == 0:
                self.r2dEdw[i] = np.outer(self.r2dEdx[i], self.x)
            else:
                self.r2dEdw[i] = np.outer(self.r2dEdx[i], self.yhat[i - 1])
                self.r2dEdw[i] += 2.0 * np.outer(self.rdEdx[i], self.ry[i - 1])
                self.r2dEdw[i] += np.outer(self.dEdx[i], self.r2y[i - 1])

    def compute(self):
        self.forward()
        self.backward()

    def compare(self):

        # check if compute performed
        if not self.r2dEdw:
            raise Exception('Run compute before compare.')

        # reformat
        vghv = []
        for i in range(0, self.len):
            vghv.extend(self.r2dEdw[i].flatten(order='F'))

        vghvMat = self.mdict.get('vghv')  # get value from mdict
        vghvMat = np.reshape(vghvMat, np.shape(vghv), order='F')

        if vghvMat is None:
            raise Exception('vghv does not exist in Matlab data file.')

        diff = vghvMat-vghv  # compute difference
        norm = np.linalg.norm(diff)  # compute norm

        return norm

    def ropCompare(self):

        # check if compute performed
        if not self.rdEdw:
            raise Exception('Run compute before compare.')

        # reformat
        hv = []
        for i in range(0, self.len):
            hv.extend(self.rdEdw[i].flatten(order='F'))

        hvMat = self.mdict.get('hv')  # get value from mdict
        hvMat = np.reshape(hvMat, np.shape(hv), order='F')

        if hvMat is None:
            raise Exception('hv does not exist in Matlab data file.')

        diff = hvMat-hv  # compute difference
        norm = np.linalg.norm(diff)  # compute norm

        return norm

    def gradCompare(self):

        # check if compute performed
        if not self.rdEdw:
            raise Exception('Run compute before compare.')

        # reformat
        g = []
        for i in range(0, self.len):
            g.extend(self.dEdw[i].flatten(order='F'))

        gMat = self.mdict.get('g')  # get value from mdict
        gMat = np.reshape(gMat, np.shape(g), order='F')

        if gMat is None:
            raise Exception('g does not exist in Matlab data file.')

        diff = gMat-g  # compute difference
        norm = np.linalg.norm(diff)  # compute norm

        return norm
