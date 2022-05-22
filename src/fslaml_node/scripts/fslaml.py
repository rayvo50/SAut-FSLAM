#!/usr/bin/env python3

import time
import sys
import numpy as np
from math import sqrt

import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


class Slamings():
    def __init__(self, info_pub) -> None:
        self
    
    def callback(self, imu, odom):
        rospy.loginfo("IMU DATA:\n")
        rospy.loginfo(imu)
        rospy.loginfo("ODOM DATA:\n")
        rospy.loginfo(odom)
        



def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    # map_pub = rospy.Publisher('detected_cloud', PointCloud2, queue_size=2)

    slammings = Slamings(info_pub)

    ats = ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.3, allow_headerless=False)
    ats.registerCallback(slammings.callback)

    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass