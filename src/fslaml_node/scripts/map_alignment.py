#!/usr/bin/env python3

from operator import ne
import sys
import numpy as np
from math import sqrt, pi, cos, sin, atan2, floor

import matplotlib.pyplot as plt
import rospy
import cv2 
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String, Header
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from tf.transformations import euler_from_quaternion
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

from proscrutes import *


# micro simulation
ROOM_SIZE = 5
MAP = np.array([
    [2,2],
    [3,-4],
    [-3, 2],
    [1,-1],
    [-1,-2],
    [2,3],
    [3,-1],
    [-1,0],
    [0,-2],
    [-1,1]
    ])
CAM_FOV = 90
COLORS = [(0,0,255/255),(0,128/255,255/255),(0,255/255,255/255),(128/255,128/255,128/255),(0,255/255,0),(255/255,255/255,0),(255/255,128/255,0),(255/255,0,0),(255/255,0,128/255),(255/255,0,255/255)]

def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

# Utility function to draw  results
def draw_m_2_px(img, map, pose):
    pose = (-1*floor(pose[1]*100) + 500, -1*floor(pose[0]*100) +500)
    cv2.circle(img, pose, 2, (0,153,76), cv2.FILLED)
    map = map.reshape(-1, 2)
    for lm in map:
        lm_center = (-1*floor(lm[1]*100)+500, -1*floor(lm[0]*100)+500)
        cv2.circle(img, lm_center, 2, (0, 128, 255), cv2.FILLED)

# Main class for implementing ROS stuff
class Mapper():
    def __init__(self, pub):
        self.pub = pub
        self.id = -1
        self.map = []
        self.ok = 0

    def callback(self, map_msg):
        self.map_data = map_msg
        self.ok = 1


    def process(self):
        if not self.ok:
            return
        id = int(self.map_data.header.frame_id)
        if id == self.id:
            pass # plot cenas 

        x = self.map_data.poses[0].position.x
        y = self.map_data.poses[0].position.y
        quat = self.map_data.poses[0].orientation
        euler = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        teta = euler[2]
        
        new_pose = (x, y, teta)

        new_map = []
        for i in range(1, len(self.map_data.poses)):
            new_map.append([self.map_data.poses[i].position.x, self.map_data.poses[i].position.y, int(self.map_data.poses[i].orientation.x)])

        if len(self.map) == 0:
            self.map = new_map
            self.pose = new_pose
            return

        pose1 = np.array([self.pose[0], self.pose[1]])
        pose2 = np.array([new_pose[0], new_pose[1]])

        data1 = np.array(self.map)
        data2 = np.array(new_map)
        while len(data1) != len(data2):
            if len(data1) > len(data2):
                print("shit")
            data1 = np.vstack([data1, [0,0]])
    
        
        mtx1 = np.array(data1, dtype=np.double, copy=True)
        mtx2 = np.array(data2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        trans1 = np.mean(mtx1, 0)
        trans2 = np.mean(mtx2, 0)
        mtx1 -= trans1
        mtx2 -= trans2
        pose1 -= trans1
        pose2 -= trans2

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = orthogonal_procrustes(mtx1, mtx2)
        mtx2 = np.dot(mtx2, R.T) * s
        pose2 = np.dot(pose2, R.T) * s

        mtx1 *= norm1
        mtx2 *= norm2
        mtx1 += trans1
        mtx2 += trans1
        pose1 += trans1
        pose2 += trans1

        self.map = np.copy(mtx2)
        plot2(mtx2, pose2)

    
    def plot(self):
        plt.clf()
        if len(self.map) == 0:
            return
        x = self.map[:, 0]
        y = self.map[:, 1]
        plt.plot(-1*y, x, 'x')
        #plt.plot(-1*self.pose[1], self.pose[0], 'o')
        plt.draw()        
        plt.pause(0.00000000001)


def plot1(map):
        plt.clf()
        if len(map) == 0:
            return
        x = map[:, 0]
        y = map[:, 1]
        plt.plot(-1*y, x, 'x')
        #plt.plot(-1*self.pose[1], self.pose[0], 'o')
        plt.draw()        
        plt.pause(0.00000000001)

def plot2(old_map, new_map):
    plt.clf()
    map = old_map
    if len(map) == 0:
        return
    x = map[:, 0]
    y = map[:, 1]
    plt.plot(-1*y, x, 'x')

    map = new_map
    if len(map) == 0:
        return
    x = map[0]
    y = map[1]
    plt.plot(-1*y, x, 'o')
    #plt.plot(-1*self.pose[1], self.pose[0], 'o')
    plt.draw()
    plt.pause(0.00000000001)


def main(args):

    rospy.init_node('map_alignment', anonymous=True)
    rospy.loginfo('Initializing map alignment node with Python version: ' + sys.version)

    map_sub = Subscriber('fast_slam_map', PoseArray)
    
    info_pub = rospy.Publisher('info', String, queue_size=2)

    m = Mapper(info_pub)
    rate = rospy.Rate(2)
    ats = ApproximateTimeSynchronizer([map_sub], queue_size=10, slop=0.01, allow_headerless=False)
    ats.registerCallback(m.callback)

    while not rospy.is_shutdown():
        m.process()
        rate.sleep()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass