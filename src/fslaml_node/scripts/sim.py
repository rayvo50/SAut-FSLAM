#!/usr/bin/env python3

import time
import sys
import numpy as np
from math import sqrt, pi, cos, sin
from numpy.random import uniform


class LandmarkEKF():
    def __init__(self, mean, sigma, noise_teta, noise_d) -> None:
        self.mean = np.array(np.reshape(mean, (2,1)))  # [mean_x, mean_y]
        self.sigma = np.array(np.reshape(sigma, (2,2)))   # covariance matrix 
        self.Qt = np.array([[noise_d, 0],[0, noise_teta]]) # noise in matrix form

    def update(self, d, teta):
        # predict part (in this case doesnt change anything?)
        ut = np.reshape(self.mean, (2,1)) 
        sigmat = self.sigma
        # update part
        Ht = np.array(
            [[self.mean[0][0]/sqrt(self.mean[0][0]**2 + self.mean[1][0]**2), 
            self.mean[1][0]/sqrt(self.mean[0][0]**2 + self.mean[1][0]**2)],
            [-self.mean[1][0]/(self.mean[0][0]**2 + self.mean[1][0]**2),
            1/(self.mean[0][0] + ((self.mean[1][0]**2)/self.mean[0][0]))]])
        Ht = np.reshape(Ht, (2,2))
        #Ht = np.array(
            # [[self.mean[0]/sqrt(self.mean[0]**2 + self.mean[1]**2), 
            # self.mean[0]/sqrt(self.mean[1]**2 + self.mean[1]**2)],
            # [-self.mean[1]/(self.mean[0]**2 + self.mean[1]**2),
            # 1/(self.mean[0] + ((self.mean[1]**2)/self.mean[0]))]])
        #Kt = self.sigma * Ht.getT() * np.linalg.inv(Ht * self.sigma * Ht.getT() + self.Qt)  
        temp = Ht @ sigmat @ Ht.T + self.Qt
        temp = np.reshape(temp, (2,2))
        Kt = sigmat @ Ht.T @ np.linalg.inv(temp)
        zt = np.reshape([d, teta], (2,1)) 
        # here need to check if it between 0 and 2pi, still need to make that func xd nad change all over the code for using it
        self.mean = ut + Kt @ (zt - Ht @ ut)
        self.sigma = (np.array([[1,0], [0,1]]) - Kt @ Ht) @ sigmat


def main(args):
    x = 2
    y = 2
    ldmrks = []
    ldmrks.append(LandmarkEKF())


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass