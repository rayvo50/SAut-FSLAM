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

M_PARTICLES = 1000

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
        #p = uniform((-10, -10, -pi/2),(10, 10, pi/2),3)
        p = uniform((-0.5, -0.5, -0.01),(0.5, 0.5, 0.01),3)
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
        self.x = 0
        self.y = 0
        self.teta = 0  
    
    def show_particle_state(self): #TODO add orientation to this representation so it looks nicer
        img = np.zeros((700,700,3), dtype=np.uint8)
        for p in self.Xt:
            cv2.circle(img, (round(int(p.x)*35) + 350, round(int(p.y)*35) + 350), 3, (255, 0, 0), cv2.FILLED)
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
            self.teta += delta_teta
            self.x += delta_x*cos(self.teta)
            self.y += delta_x*sin(self.teta)
            self.pub.publish(str(self.x) + "\n" + str(self.y) + "\n" + str(self.teta) + "\n")
            # for i in range(len(self.Xt)):
            #     self.Xt[i].teta += delta_teta
            #     self.Xt[i].x += delta_x*cos(self.Xt[i].teta)
            #     self.Xt[i].y += delta_x*sin(self.Xt[i].teta)
            # #calculate weights    
            # self.show_particle_state()



def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles', Image, queue_size=2)
    # map_pub = rospy.Publisher('detected_cloud', PointCloud2, queue_size=2)

    pf = ParticleFilter(info_pub, image_pub)

    ats = ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.3, allow_headerless=False)
    ats.registerCallback(pf.callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass