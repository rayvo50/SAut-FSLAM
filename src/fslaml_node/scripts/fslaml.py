#!/usr/bin/env python3

import time
import sys
import numpy as np
from math import sqrt, pi, cos, sin
from numpy.random import uniform


import rospy
import cv2 
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry


SHOW = 0
try:
    import matplotlib.pyplot as plt
    SHOW = 1
except BaseException as error:
    pass

M_PARTICLES = 100
ROOM_SIZE = 4

# Defines the shape of the particles
class Particle():
    def __init__(self, x, y, teta, ldmrks, w):
        self.x = x
        self.y = y
        self.teta = teta
        self.ldmrks = ldmrks
        self.w = w

# Utility for manipulating particle sets 
class Particle_set():
    def __init__(self) -> None:
        pass
    
    def gen_random(self):
        #p = uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
        p = np.zeros(3)     # assume boot satarting location as inertial origin
        ldmrks = []
        return Particle(p[0], p[1], p[2], ldmrks, 1)

# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, image_pub) -> None:
        self.pub = info_pub
        self.img_pub = image_pub
        self.bridge = CvBridge()
        self.prev_t = 0
        p_set = Particle_set()
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(p_set.gen_random())
            self.Xt[i].w = 1/M_PARTICLES

        #TODO: this is an image representaation, some kind of plot would be better, força malucos
    def show_particle_state(self): #TODO add orientation to this representation so it looks nicer
        img = np.zeros((700,700,3), dtype=np.uint8)
        for p in self.Xt:
            center = (int(p.x*210) + 350, int(p.y*70) +350)
            cv2.circle(img, center, 4, (0, 255, 0), cv2.FILLED)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))


    def callback(self, odom, imu):
        #calculate shift in particle location based on odometry data
        if not(self.prev_t): # if there is no previous time
            self.prev_t = odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000
        else:
            curr_t = odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000
            timing = (curr_t - self.prev_t)/1000000000
            self.prev_t = curr_t
            delta_teta = odom.twist.twist.angular.z * timing
            delta_x = odom.twist.twist.linear.x * timing
            for i in range(len(self.Xt)):
                self.Xt[i].teta += delta_teta + np.random.normal(0, abs(delta_teta/(2*1.645))) # delta
                self.Xt[i].x += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*cos(self.Xt[i].teta)
                self.Xt[i].y += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*sin(self.Xt[i].teta)
            

            #calculate weights    
            self.show_particle_state()



def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles', Image, queue_size=2)

    pf = ParticleFilter(info_pub, image_pub)
    pf.show_particle_state()

    ats = ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.3, allow_headerless=False)
    ats.registerCallback(pf.callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass