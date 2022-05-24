#!/usr/bin/env python3

from dataclasses import dataclass
import time
import sys
import numpy as np
from math import sqrt, pi
from numpy.random import uniform
from dataclasses import dataclass


import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


SHOW = 0
try:
    import matplotlib.pyplot as plt
    SHOW = 1
except BaseException as error:
    pass

M_PARTICLES = 100

# Defines the shape of the particles
class Particle():
    def __init__(self, x, y, teta, ldmrks ):
        self.x = x
        self.y = y
        self.teta = teta
        self.ldmrks = ldmrks

# Utility for manipulating particle sets 
class Particle_set():
    def __init__(self) -> None:
        pass
    
    def gen_random(self):
        p = uniform((-10, -10, -pi/2),(10, 10, pi/2),3)
        ldmrks = []
        return Particle(p[0], p[1], p[2], ldmrks)

# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub) -> None:
        self.pub = info_pub
        p_set = Particle_set()
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(p_set.gen_random())
    
    def show_particle_state(self): #TODO add orientation to this representation so it looks nicer
        if 1:
            x = []
            y = []
            for p in self.Xt:
                x.append(p.x)
                y.append(p.y)
            print(x)
            plt.plot(x, y, 'o')
            plt.show()
        
    def callback(self, imu, odom):
        pass

        



def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    # map_pub = rospy.Publisher('detected_cloud', PointCloud2, queue_size=2)

    pf = ParticleFilter(info_pub)
    ##print(pf.Xt)
    pf.show_particle_state


    ats = ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.3, allow_headerless=False)
    ats.registerCallback(pf.callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass