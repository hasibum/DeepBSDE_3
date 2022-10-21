import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from scipy.integrate import simps
import math


class FBSDE(object):
    """Base class for defining PDE related function."""

    def __init__(self, T,L,nbStep,num_hiddens, dtype, batch_size, valid_size, num_iterations,lp,rp,al, delt,gam):

        self.dataType=dtype
        self.T = T

        self.nbStep=nbStep
        self.TStep = self.T / self.nbStep

        self.y_init = None
        self.L=L
        self.lp=lp
        self.rp=rp
        self.h=(self.rp-self.lp)/(self.L+1)
        self.xi=np.linspace(self.lp,self.rp,self.L+2)
        self.intgStep=10
        self.alpha=al
        self.delta=delt
        self.gamma=gam


    # Finite element Setup for 1D case
    def MUt(self):
        return tf.tensordot(tf.linalg.inv(self.A()),self.B(),axes=1)

    def A(self):
        temp=((2/3)*np.eye(self.L,k=0)+(1/6)*np.eye(self.L,k=-1)+(1/6)*np.eye(self.L,k=1))*self.h
        return temp
        #return tf.convert_to_tensor(temp,dtype=self.dataType)

    def B(self):
        temp=(2*np.eye(self.L,k=0)-1*np.eye(self.L,k=-1)-1*np.eye(self.L,k=1))/self.h
        return temp
        #return tf.convert_to_tensor(temp,dtype=self.dataType)

    def M(self):
        temp=-np.eye(self.L,k=1)/2+np.eye(self.L,k=-1)/2
        return temp
        #return tf.convert_to_tensor(temp,dtype=self.dataType)


    def sample(self, num_sample):
        xini = np.einsum("ij,j->i", np.linalg.inv(self.A()), self.x0phi())
        x0= np.tile(xini, [num_sample, 1])
        rand = np.sqrt(self.TStep) * np.random.normal(size=[self.nbStep, 1, 1])
        for s in range(num_sample - 1):
            rand = np.append(rand, np.sqrt(self.TStep) * np.random.normal(size=[self.nbStep, 1, 1]), axis=1)
        dw_sample=rand
        return tf.convert_to_tensor(dw_sample), tf.convert_to_tensor(x0)


    def g(self, r):
        return tf.math.atan(r)

    def fncl(self, xp, x):
        return (x - xp) / self.h

    def fncr(self, xp, x):
        return (xp - x) / self.h

    def dfncl(self):
        return 1 / self.h

    def dfncr(self):
        return -1 / self.h

    def YT(self, xV, lbp, rbp):
        a = tf.tensordot(self.funphi(self.T, xV, self.g, lbp, rbp), tf.linalg.inv(self.A()), [[1], [1]])
        # a = tf.einsum("ij,kj->ki",tf.linalg.inv(self.A()),self.funphi(self.T,xV, bs,self.g))
        return a

        # Simpson numerical batch integration with y[batch,d] ,x[d]

    def integral(self, y, x):
        dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
        return (y[:, 0] + y[:, -1] + 4 * tf.reduce_sum(y[:, 1:-1:2], 1) + 2 * tf.reduce_sum(y[:, 2:-1:2], 1)) * dx / 3

    def F(self,  X):
        return -tf.tensordot(X, self.delta * tf.convert_to_tensor(tf.transpose(self.MUt()), dtype=self.dataType), 1)

    def F2(self, ntstep, X, Y, Wt, xpoints):
        return tf.tensordot(self.f2phi(ntstep, X, Y, Wt, xpoints), tf.linalg.inv(self.A()), [[1], [1]])
        # return np.einsum("kj,ij->ki",self.f2phi(ntstep, X, Y, bs, dw,xpoints),np.linalg.inv(self.A()))

    def F1(self, ntstep, X, Y, Wt, xpoints):
        return self.TStep * tf.tensordot(self.f1phi(ntstep, X, Y, Wt, xpoints), tf.linalg.inv(self.A()),[[1], [1]])  # einsum("kj,ij->ki"

    def F3(self, Wt, xpoints, dWt):
        return tf.tensordot(self.f3phi(Wt, xpoints), tf.linalg.inv(self.A()), [[1], [1]]) * dWt  # einsum("kj,ij->ki"

    def x0phi(self):
        X0 = []
        i = 0
        while (i < self.L):
            X0_i_1 = (-np.sin(np.pi * self.xi[i]) + 2 * np.sin(np.pi * self.xi[i + 1]) - np.sin(
                np.pi * self.xi[i + 2])) / (2 * self.h * np.pi)
            X0_i_2 = (-np.sin(2 * np.pi * self.xi[i]) + 2 * np.sin(2 * np.pi * self.xi[i + 1]) - np.sin(
                2 * np.pi * self.xi[i + 2])) / (8 * self.h * (np.pi ** 2))
            X0_i = X0_i_1 + X0_i_2
            X0.append(X0_i)
            i = i + 1
        return X0
        #return tf.convert_to_tensor(X0, dtype=self.dataType)



####################################
    def generateX1stepFw(self,t_n, xt_n, yt_n,Wt_n, xpoints,dWt_n):
        temp = tf.linalg.inv(np.eye(self.L) + self.delta*self.MUt() * self.TStep)
        xtn_1=tf.einsum("ij,kj->ki", tf.convert_to_tensor(temp,dtype=self.dataType), xt_n + self.F1(t_n,xt_n,yt_n,Wt_n,xpoints)-self.F3(Wt_n,xpoints,dWt_n))
        return xtn_1




    def funphi(self,t,xV,g2,lbp,rbp):
        g2p=[]
        x1 = tf.convert_to_tensor(np.linspace(self.xi[0], self.xi[1], self.intgStep), dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[1], self.xi[2], self.intgStep), dtype=self.dataType)
        y1=g2(tf.tile(tf.expand_dims(xV[:, 0],1),[1,self.intgStep])*lbp)*lbp
        y2=g2(tf.tile(tf.expand_dims(xV[:, 0],1),[1,self.intgStep])*rbp+tf.tile(tf.expand_dims(xV[:, 1],1),[1,self.intgStep])*lbp)*rbp
        g2p.append(self.integral(y1,x1)+self.integral(y2,x2))
        l=2
        while(l<=self.L-1):
            x1 = tf.convert_to_tensor(np.linspace(self.xi[l - 1], self.xi[l], self.intgStep), dtype=self.dataType)
            x2 = tf.convert_to_tensor(np.linspace(self.xi[l], self.xi[l + 1], self.intgStep), dtype=self.dataType)
            y1=g2(tf.tile(tf.expand_dims(xV[:, l-2],1),[1,self.intgStep])*rbp+tf.tile(tf.expand_dims(xV[:, l-1],1),[1,self.intgStep])*lbp)*lbp
            y2=g2(tf.tile(tf.expand_dims(xV[:, l-1],1),[1,self.intgStep])*rbp+tf.tile(tf.expand_dims(xV[:, l],1),[1,self.intgStep])*lbp)*rbp
            g2p.append(self.integral(y1, x1) + self.integral(y2, x2))
            l=l+1
        x1 = tf.convert_to_tensor(np.linspace(self.xi[self.L - 1], self.xi[self.L], self.intgStep), dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[self.L], self.xi[self.L + 1], self.intgStep), dtype=self.dataType)
        y1=g2(tf.tile(tf.expand_dims(xV[:, self.L-2],1),[1,self.intgStep])*rbp+tf.tile(tf.expand_dims(xV[:, self.L-1],1),[1,self.intgStep])*lbp)*lbp
        y2=g2(tf.tile(tf.expand_dims(xV[:, self.L-1],1),[1,self.intgStep])*rbp)*rbp
        g2p.append(self.integral(y1, x1) + self.integral(y2, x2))
        return tf.transpose(g2p)

    def f1phi(self,ntstep,xV,yV,Wt,xpoints):
        pi=tf.constant(math.pi,dtype=self.dataType)
        f1dp=[]
        x1 = tf.convert_to_tensor(np.linspace(self.xi[0], self.xi[1], self.intgStep),dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[1], self.xi[2], self.intgStep),dtype=self.dataType)
        rho1=tf.einsum("i,j->ij", xV[:, 0], self.fncl(self.xi[0], x1))
        u1=tf.einsum("i,j->ij", yV[:, 0], self.fncl(self.xi[0], x1))
        rho2=tf.einsum("i,j->ij", xV[:, 0], self.fncr(self.xi[2], x2)) + tf.einsum("i,j->ij",xV[:, 1],self.fncl(self.xi[1],x2))
        u2=tf.einsum("i,j->ij", yV[:, 0], self.fncr(self.xi[2], x2))+ tf.einsum("i,j->ij",yV[:, 1],self.fncl(self.xi[1],x2))
        y11=self.alpha*tf.math.cos(u1)-self.alpha*(1/tf.sqrt(1+tf.square(rho1)))+self.delta*tf.square(pi)*rho1\
            +self.delta*(2+tf.math.cos(tf.tile(Wt,[1,self.intgStep])))*(pi**2)*tf.math.sin(2*pi*xpoints[0])/2\
            -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[0])/12
        y1=tf.einsum("ij,j->ij",y11, self.fncl(self.xi[0], x1))
        y12=self.alpha*tf.math.cos(u2)-self.alpha*(1/tf.sqrt(1+tf.square(rho2)))+self.delta*tf.square(pi)*rho2\
            +self.delta*(2+tf.math.cos(tf.tile(Wt,[1,self.intgStep])))*(pi**2)*tf.math.sin(2*pi*xpoints[1])/2\
            -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[1])/12
        y2=tf.einsum("ij,j->ij",y12 , self.fncr(self.xi[2], x2))
        f1dp.append(self.integral(y1, x1) + self.integral(y2, x2))
        l=2
        while (l <= self.L - 1):
            x1 = tf.convert_to_tensor(np.linspace(self.xi[l-1], self.xi[l], self.intgStep), dtype=self.dataType)
            x2 = tf.convert_to_tensor(np.linspace(self.xi[l], self.xi[l+1], self.intgStep), dtype=self.dataType)
            rho1 = tf.einsum("i,j->ij",xV[:, l-2],self.fncr(self.xi[l], x1))+tf.einsum("i,j->ij",xV[:, l-1],self.fncl(self.xi[l-1], x1))
            u1 = tf.einsum("i,j->ij",yV[:, l-2],self.fncr(self.xi[l], x1))+tf.einsum("i,j->ij",yV[:, l-1],self.fncl(self.xi[l-1], x1))
            rho2 = tf.einsum("i,j->ij",xV[:, l-1],self.fncr(self.xi[l+1], x2))+tf.einsum("i,j->ij",xV[:, l],self.fncl(self.xi[l],x2))
            u2 = tf.einsum("i,j->ij",yV[:, l-1],self.fncr(self.xi[l+1], x2))+tf.einsum("i,j->ij",yV[:, l],self.fncl(self.xi[l],x2))
            y11 = self.alpha*tf.math.cos(u1) - self.alpha*(1 / tf.sqrt(1 + tf.square(rho1))) + self.delta*tf.square(pi) * rho1 \
                  + self.delta*(2 +  tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * (pi ** 2) * tf.math.sin(2 * pi * xpoints[l-1]) / 2\
                  -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[l-1])/12
            y1 = tf.einsum("ij,j->ij", y11, self.fncl(self.xi[l-1], x1))
            y12 = self.alpha*tf.math.cos(u2) - self.alpha*(1 / tf.sqrt(1 + tf.square(rho2))) + self.delta*tf.square(pi) * rho2 \
                  + self.delta*(2 +  tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * (pi ** 2) * tf.math.sin(2 * pi * xpoints[l]) / 2\
                  -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[l])/12
            y2 = tf.einsum("ij,j->ij", y12, self.fncr(self.xi[l+1], x2))
            f1dp.append(self.integral(y1, x1) + self.integral(y2, x2))
            l = l + 1
        x1 = tf.convert_to_tensor(np.linspace(self.xi[self.L-1], self.xi[self.L], self.intgStep), dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[self.L], self.xi[self.L+1], self.intgStep), dtype=self.dataType)
        rho1=tf.einsum("i,j->ij",xV[:, self.L - 2],self.fncr(self.xi[self.L], x1))+tf.einsum("i,j->ij",xV[:, self.L-1],self.fncl(self.xi[self.L-1],x1))
        u1=tf.einsum("i,j->ij",yV[:, self.L - 2],self.fncr(self.xi[self.L], x1))+tf.einsum("i,j->ij",yV[:, self.L-1],self.fncl(self.xi[self.L-1],x1))
        rho2=tf.einsum("i,j->ij",xV[:, self.L-1],self.fncr(self.xi[self.L+1], x2))
        u2=tf.einsum("i,j->ij",yV[:, self.L-1],self.fncr(self.xi[self.L+1], x2))
        y11 = self.alpha*tf.math.cos(u1) - self.alpha*(1 / tf.sqrt(1 + tf.square(rho1))) + self.delta*tf.square(pi) * rho1 \
              + self.delta*(2 +  tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * (pi ** 2) * tf.math.sin(2 * pi * xpoints[self.L - 1]) / 2\
              -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[self.L-1])/12
        y1 = tf.einsum("ij,j->ij", y11, self.fncl(self.xi[self.L - 1], x1))
        y12 = self.alpha*tf.math.cos(u2) - self.alpha*(1 / tf.sqrt(1 + tf.square(rho2))) + self.delta*tf.square(pi) * rho2 \
              + self.delta*(2 +  tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * (pi ** 2) * tf.math.sin(2 * pi * xpoints[self.L]) / 2\
              -tf.math.cos(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[self.L])/12
        y2 = tf.einsum("ij,j->ij", y12, self.fncr(self.xi[self.L + 1], x2))

        f1dp.append(self.integral(y1, x1) + self.integral(y2, x2))
        return tf.transpose(f1dp)

    def f3phi(self,Wt,xpoints):
        pi=tf.constant(math.pi,dtype=self.dataType)
        # Wt has shape (bs,1) All componenets get similar BM

        f3p=[]
        x1 = tf.convert_to_tensor(np.linspace(self.xi[0], self.xi[1], self.intgStep),dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[1], self.xi[2], self.intgStep),dtype=self.dataType)
        y11=tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[0])/6
        y1=tf.einsum("ij,j->ij",y11, self.fncl(self.xi[0], x1))
        y12=tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[1])/6
        y2=tf.einsum("ij,j->ij",y12 , self.fncr(self.xi[2], x2))
        f3p.append(self.integral(y1, x1) + self.integral(y2, x2))
        l=2
        while (l <= self.L - 1):
            x1 = tf.convert_to_tensor(np.linspace(self.xi[l-1], self.xi[l], self.intgStep), dtype=self.dataType)
            x2 = tf.convert_to_tensor(np.linspace(self.xi[l], self.xi[l+1], self.intgStep), dtype=self.dataType)
            y11 = tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[l-1])/6
            y1 = tf.einsum("ij,j->ij", y11, self.fncl(self.xi[l-1], x1))
            y12 = tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[l])/6
            y2 = tf.einsum("ij,j->ij", y12, self.fncr(self.xi[l+1], x2))
            f3p.append(self.integral(y1, x1) + self.integral(y2, x2))
            l = l + 1
        x1 = tf.convert_to_tensor(np.linspace(self.xi[self.L-1], self.xi[self.L], self.intgStep), dtype=self.dataType)
        x2 = tf.convert_to_tensor(np.linspace(self.xi[self.L], self.xi[self.L+1], self.intgStep), dtype=self.dataType)
        y11 = tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[self.L-1])/6
        y1 = tf.einsum("ij,j->ij", y11, self.fncl(self.xi[self.L - 1], x1))
        y12 = tf.math.sin(tf.tile(Wt,[1,self.intgStep]))*tf.math.sin(2*pi*xpoints[self.L])/6
        y2 = tf.einsum("ij,j->ij", y12, self.fncr(self.xi[self.L + 1], x2))
        f3p.append(self.integral(y1, x1) + self.integral(y2, x2))
        return tf.transpose(f3p)


    def f2phi(self, ntstep, xV, yV, Wt,xpoints):
        pi = tf.constant(math.pi,dtype=self.dataType)
        Intg = self.It(xV, xpoints)
        y5 = self.gamma*Intg - self.gamma*(2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) / 12
        # Wt has shape (bs,1) All componenets get similar BM
        f2p=[]
        x1 = tf.linspace(self.xi[0], self.xi[1], self.intgStep)
        x2 = tf.linspace(self.xi[1], self.xi[2], self.intgStep)
        rho1 = tf.einsum("i,j->ij", xV[:, 0], self.fncl(self.xi[0], x1))
        u1 = tf.einsum("i,j->ij", yV[:, 0], self.fncl(self.xi[0], x1))
        rho2 = tf.einsum("i,j->ij", xV[:, 0], self.fncr(self.xi[2], x2)) + tf.einsum("i,j->ij", xV[:, 1],self.fncl(self.xi[1], x2))
        u2 = tf.einsum("i,j->ij", yV[:, 0], self.fncr(self.xi[2], x2))  + tf.einsum("i,j->ij", yV[:, 1],self.fncl(self.xi[1], x2))
        y11=self.delta*(2*rho1/tf.square(1+tf.square(rho1)))*tf.square(tf.square(pi)*tf.math.cos(pi*xpoints[0])/2+(2+tf.math.cos(tf.tile(Wt,[1, self.intgStep]))*pi*tf.math.cos(2*pi*xpoints[0]))/3)
        y21=self.delta*(2/(1+tf.square(rho1)))*((pi**3)*tf.math.sin(pi*xpoints[0])/2+(2+tf.math.cos(tf.tile(Wt,[1, self.intgStep])))*2*tf.square(pi)*tf.math.sin(2*pi*xpoints[0])/3)
        y31=(rho1/tf.square(1+tf.square(rho1)))*tf.square(tf.math.sin(tf.tile(Wt,[1, self.intgStep])))*tf.square(tf.math.sin(2*pi*xpoints[0]))/36
        y41 = (1 / (1 + tf.square(rho1))) * (tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[0]) / 12 - self.delta*tf.square(pi) * rho1
                    - self.delta*((2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(2 * pi * xpoints[0]) / 2))
        y51=self.alpha*u1-self.alpha*tf.math.atan(rho1)
        y1 = tf.einsum("ij,j->ij", y11+y21+y31+y41+y5+y51, self.fncl(self.xi[0], x1))
        y12=self.delta*(2*rho2/tf.square(1+tf.square(rho2)))*tf.square(tf.square(pi)*tf.math.cos(pi*xpoints[1])/2+(2+tf.math.cos(tf.tile(Wt,[1, self.intgStep]))*pi*tf.math.cos(2*pi*xpoints[1]))/3)
        y22=self.delta*(2/(1+tf.square(rho2)))*((pi**3)*tf.math.sin(pi*xpoints[1])/2+(2+tf.math.cos(tf.tile(Wt,[1, self.intgStep])))*2*tf.square(pi)*tf.math.sin(2*pi*xpoints[1])/3)
        y32=(rho2/tf.square(1+tf.square(rho2)))*tf.square(tf.math.sin(tf.tile(Wt,[1, self.intgStep])))*tf.square(tf.math.sin(2*pi*xpoints[1]))/36
        y42 = (1 / (1 + tf.square(rho2))) * (tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[1]) / 12
                    - self.delta*tf.square(pi) * rho2 - self.delta*((2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(2 * pi * xpoints[1]) / 2))
        y52 = self.alpha * u2 - self.alpha * tf.math.atan(rho2)
        y2 = tf.einsum("ij,j->ij", y12+y22+y32+y42+y5+y52, self.fncr(self.xi[2], x2))
        f2p.append(self.integral(y1, x1) + self.integral(y2, x2))
        l=2
        while (l <= self.L - 1):
            x1 = tf.linspace(self.xi[l - 1], self.xi[l], self.intgStep)
            x2 = tf.linspace(self.xi[l], self.xi[l + 1], self.intgStep)
            rho1 = tf.einsum("i,j->ij", xV[:, l - 2], self.fncr(self.xi[l], x1)) + tf.einsum("i,j->ij", xV[:, l - 1],self.fncl(self.xi[l - 1],x1))
            u1 = tf.einsum("i,j->ij", yV[:, l - 2], self.fncr(self.xi[l], x1)) + tf.einsum("i,j->ij",yV[:, l - 1],self.fncl(self.xi[l - 1],x1))
            rho2 = tf.einsum("i,j->ij", xV[:, l - 1], self.fncr(self.xi[l + 1], x2)) + tf.einsum("i,j->ij", xV[:, l],self.fncl(self.xi[l],x2))
            u2 = tf.einsum("i,j->ij", yV[:, l - 1],  self.fncr(self.xi[l + 1], x2)) + tf.einsum("i,j->ij",yV[:, l],self.fncl(self.xi[l],x2))
            y11 = self.delta*(2 * rho1 / tf.square(1 + tf.square(rho1))) * tf.square(tf.square(pi) * tf.math.cos(pi * xpoints[l-1]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * pi * tf.math.cos(2 * pi * xpoints[l-1])) / 3)
            y21 = self.delta*(2 / (1 + tf.square(rho1))) * ((pi ** 3) * tf.math.sin(pi * xpoints[l-1]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * 2 * tf.square(pi) * tf.math.sin(2 * pi * xpoints[l-1]) / 3)
            y31 = (rho1 / tf.square(1 + tf.square(rho1))) * tf.square(tf.math.sin(tf.tile(Wt, [1, self.intgStep]))) * tf.square(tf.math.sin(2 * pi * xpoints[l-1])) / 36
            y41 = (1 / (1 + tf.square(rho1))) * (
                        tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[l - 1]) / 12
                        - self.delta*tf.square(pi) * rho1 - self.delta*((2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(2 * pi * xpoints[l - 1]) / 2))
            y51 = self.alpha * u1 - self.alpha * tf.math.atan(rho1)
            y1 = tf.einsum("ij,j->ij", y11+y21+y31+y41+y5+y51, self.fncl(self.xi[l - 1], x1))
            y12 = self.delta*(2 * rho2 / tf.square(1 + tf.square(rho2))) * tf.square(tf.square(pi) * tf.math.cos(pi * xpoints[l]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * pi * tf.math.cos(2 * pi * xpoints[l])) / 3)
            y22 = self.delta*(2 / (1 + tf.square(rho2))) * ((pi ** 3) * tf.math.sin(pi * xpoints[l]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * 2 * tf.square(pi) * tf.math.sin(2 * pi * xpoints[l]) / 3)
            y32 = (rho2 / tf.square(1 + tf.square(rho2))) * tf.square(tf.math.sin(tf.tile(Wt, [1, self.intgStep]))) * tf.square(tf.math.sin(2 * pi * xpoints[l])) / 36
            y42 = (1 / (1 + tf.square(rho2))) * (
                        tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[l]) / 12
                        - self.delta*tf.square(pi) * rho2 - self.delta*((2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(
                                2 * pi * xpoints[l]) / 2))
            y52 = self.alpha * u2 - self.alpha * tf.math.atan(rho2)
            y2 = tf.einsum("ij,j->ij", y12+y22+y32+y42+y5+y52, self.fncr(self.xi[l + 1], x2))
            f2p.append(self.integral(y1, x1) + self.integral(y2, x2))
            l = l + 1
        x1 = tf.linspace(self.xi[self.L - 1], self.xi[self.L], self.intgStep)
        x2 = tf.linspace(self.xi[self.L], self.xi[self.L + 1], self.intgStep)
        rho1 = tf.einsum("i,j->ij", xV[:, self.L - 2], self.fncr(self.xi[self.L], x1)) + tf.einsum("i,j->ij",xV[:, self.L - 1],self.fncl(self.xi[self.L - 1],x1))
        u1 = tf.einsum("i,j->ij", yV[:, self.L - 2], self.fncr(self.xi[self.L], x1)) + tf.einsum("i,j->ij",yV[:,self.L - 1],self.fncl(self.xi[self.L - 1],x1))
        rho2 = tf.einsum("i,j->ij", xV[:, self.L - 1], self.fncr(self.xi[self.L + 1], x2))
        u2 = tf.einsum("i,j->ij", yV[:, self.L - 1], self.fncr(self.xi[self.L + 1], x2))
        y11 = self.delta*(2 * rho1 / tf.square(1 + tf.square(rho1))) * tf.square(tf.square(pi) * tf.math.cos(pi * xpoints[self.L-1]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * pi * tf.math.cos(2 * pi * xpoints[self.L-1])) / 3)
        y21 = self.delta*(2 / (1 + tf.square(rho1))) * ((pi ** 3) * tf.math.sin(pi * xpoints[self.L-1]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * 2 * tf.square(pi) * tf.math.sin(2 * pi * xpoints[self.L-1]) / 3)
        y31 = (rho1 / tf.square(1 + tf.square(rho1))) * tf.square(tf.math.sin(tf.tile(Wt, [1, self.intgStep]))) * tf.square(tf.math.sin(2 * pi * xpoints[self.L-1])) / 36
        y41 = (1 / (1 + tf.square(rho1))) * (
                    tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[self.L - 1]) / 12
                    - self.delta*tf.square(pi) * rho1 - self.delta*((2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(
                            2 * pi * xpoints[self.L - 1]) / 2))
        y51 = self.alpha * u1 - self.alpha * tf.math.atan(rho1)
        y1 = tf.einsum("ij,j->ij", y11+y21+y31+y41+y5+y51, self.fncl(self.xi[self.L - 1], x1))
        y12 = self.delta*(2 * rho2 / tf.square(1 + tf.square(rho2))) * tf.square(tf.square(pi) * tf.math.cos(pi * xpoints[self.L]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * pi * tf.math.cos(2 * pi * xpoints[self.L])) / 3)
        y22 = self.delta*(2 / (1 + tf.square(rho2))) * ((pi ** 3) * tf.math.sin(pi * xpoints[self.L]) / 2 + (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * 2 * tf.square(pi) * tf.math.sin(2 * pi * xpoints[self.L]) / 3)
        y32 = (rho2 / tf.square(1 + tf.square(rho2))) * tf.square(tf.math.sin(tf.tile(Wt, [1, self.intgStep]))) * tf.square(tf.math.sin(2 * pi * xpoints[self.L])) / 36
        y42 = (1 / (1 + tf.square(rho2))) * (
                    tf.math.cos(tf.tile(Wt, [1, self.intgStep])) * tf.math.sin(2 * pi * xpoints[self.L]) / 12
                    - self.delta*tf.square(pi) * rho2 - self.delta*(
                                (2 + tf.math.cos(tf.tile(Wt, [1, self.intgStep]))) * tf.square(pi) * tf.math.sin(
                            2 * pi * xpoints[self.L]) / 2))
        y52 = self.alpha * u2 - self.alpha * tf.math.atan(rho2)
        y2 = tf.einsum("ij,j->ij", y12+y22+y32+y42+y5+y52, self.fncr(self.xi[self.L + 1], x2))
        f2p.append(self.integral(y1, x1) + self.integral(y2, x2))
        return tf.transpose(f2p)


    def xpointsintg(self, bs):
        xp=[]
        for k in range(self.L + 1):
            x = np.linspace(self.xi[k], self.xi[k + 1], self.intgStep)
            xbs = np.tile(x, [bs, 1])
            xp.append(xbs)
        return xp  #size (L+1, bs, intgstep)

    def basisxpoints(self, bs):
        xl=np.linspace(self.xi[0],self.xi[1],self.intgStep)
        xr = np.linspace(self.xi[1], self.xi[2], self.intgStep)
        Lbasispoints=np.tile(self.fncl(self.xi[0],xl),[bs,1])
        Rbasispoints = np.tile(self.fncr(self.xi[2], xr), [bs, 1])
        return Lbasispoints, Rbasispoints

 # Analytic Solution
    def u0_True(self, x):
        return np.arctan((np.pi / 2) * np.sin(np.pi * x) + np.sin(2 * np.pi * x) / 2)

    def u0xi_True(self):
        x = self.xi
        return x, self.u0_True(x)

    def u0_siml(self, Y_init):
        x=self.xi
        Y0_x=np.append([0],Y_init)
        Y0_x=np.append(Y0_x,0)
        return x, Y0_x

    def u0_simlV2(self, Y_init):
        xii = []
        for l in range(self.L + 1):
            xii.append(np.linspace(self.xi[l], self.xi[l + 1], 10))

        Y0_x = Y_init[ 0] * self.fncl(self.xi[0], xii[0])
        x = xii[0]
        l = 1
        while (l < self.L):
            Y0_x = np.append(Y0_x,
                             Y_init[l - 1] * self.fncr(self.xi[l + 1], xii[l][1:]) + Y_init[l] * self.fncl(self.xi[l], xii[l][1:]))
            x = np.append(x, xii[l][1:])
            l = l + 1
        Y0_x = np.append(Y0_x, Y_init[self.L - 1] * self.fncr(self.xi[self.L + 1], xii[self.L][1:]))
        x = np.append(x, xii[self.L][1:])
        Y0_xT=self.u0_True(x)
        return x, Y0_x, Y0_xT

    def It(self, xV,xpts):
        pi = tf.constant(math.pi, dtype=self.dataType)
        xi=tf.convert_to_tensor(self.xi,dtype=self.dataType)
        xii = []
        for l in range(self.L + 1):
            xii.append(tf.linspace(xi[l], xi[l + 1], 10))
        Intg=0
        rho1=tf.einsum("i,j->ij", xV[:, 0], self.fncl(xi[0], xii[0]))
        Intg =Intg+ self.integral(tf.math.sin(2 * pi * xpts[0]) * rho1, xii[0])


        l = 1
        while (l < self.L):
            rho1=tf.einsum("i,j->ij",xV[:,l - 1] , self.fncr(xi[l + 1], xii[l]))+ tf.einsum("i,j->ij",xV[:,l] , self.fncl(xi[l], xii[l]))
            Intg = Intg + self.integral(tf.math.sin(2 * pi * xpts[l]) * rho1, xii[l])
            l = l + 1
        rho1=tf.einsum("i,j->ij",xV[:,self.L - 1] , self.fncr(xi[self.L + 1], xii[self.L]))
        Intg = Intg + self.integral(tf.math.sin(2 * pi * xpts[self.L]) * rho1, xii[self.L])
        return tf.tile(tf.expand_dims(Intg,1),[1,self.intgStep])