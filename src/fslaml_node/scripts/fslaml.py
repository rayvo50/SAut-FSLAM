#!/usr/bin/env python3

from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import time
import sys
import numpy as np
from math import sqrt, pi, cos, sin, atan2
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

M_PARTICLES = 20
ROOM_SIZE = 5
MAP = np.array([[2,2], [3,4], [-3, -2]])
CAM_FOV = 90


def add_angle(ang1, ang2):
    res = ang1 + ang2
    if res > pi:
        return res - 2*pi
    if res < -pi:
        return res + 2*pi
    return res

# check if a given point is within a radius of a goal point
def point_is_close(goal, point, radius):
    d = sqrt(pow(goal[0]-point[0], 2) + pow(goal[1]-point[1], 2))
    return lambda d : 1 if (d < radius) else 0

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

class LandmarkEKF():
    def __init__(self, mean, sigma, noise_teta, noise_d) -> None:
        self.mean = np.array(mean) # [mean_x, mean_y]
        self.sigma = np.matrix(sigma) # [sigma_x, sigma_y]
        self.Qt = np.matrix([[noise_d, 0],[0, noise_teta]])

    def update(self, d, teta):
        # predict part (in this case doesnt change anything?)
        ut = self.mean
        sigmat = self.sigma
        # update part
        Ht = np.matrix(
            [[self.mean[0]/sqrt(self.mean[0]**2 + self.mean[1]**2), 
            self.mean[0]/sqrt(self.mean[1]**2 + self.mean[1]**2)],
            [-self.mean[1]/(self.mean[0]**2 + self.mean[1]**2),
            1/(self.mean[0] + ((self.mean[1]**2)/self.mean[0]))]])
        Kt = self.sigma * Ht.getT * np.linalg.inv(Ht * self.sigma * Ht.getT + self.Qt)
        self.mean = ut + Kt*([d, teta] - Ht*ut)
        self.sigma = (np.identity(2) - Kt*Ht)*sigmat
        rospy.loginfo("Ht:")
        rospy.loginfo(Ht)
        rospy.loginfo("Kt:")
        rospy.loginfo(Kt)
    

# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, image_pub) -> None:
        #ros stuff
        self.pub = info_pub
        self.img_pub = image_pub
        self.bridge = CvBridge()
        # particle filter stuff
        self.prev_t = 0
        self.p_set = Particle_set()
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(self.p_set.gen_random())
            self.Xt[i].w = 1/M_PARTICLES
        #for simulation (true pose of robot)
        self.x = 0
        self.y = 0
        self.teta = 0

        #TODO: this is an image representation, some kind of plot would be better, força malucos
    def draw_particles(self, img): 
        for p in self.Xt:
            pose = (int(p.x*100) + 500, int(p.y*100) +500)
            cv2.circle(img, pose, 1, (200,170,0), cv2.FILLED)
            for lm in p.ldmrks:
                lm_center = (int(lm.mean[1]*100)+500, int(lm.mean[0]*100)+500)
                cv2.circle(img, lm_center, 1, (0, 200, 255), cv2.FILLED)

    # show robot state in an 1000X1000 image, each 100px corresponds to 1 metre
    def draw_real(self, img): #TODO add orientation to this representation so it looks nicer
        true_pos = (int(self.x*100) + 500, int(self.y*100) +500)
        cv2.circle(img, true_pos, 4, (0, 255, 0), cv2.FILLED)
        for lm in MAP:
            true_lm = (int(lm[1]*100) + 500, int(lm[0]*100) +500)
            cv2.circle(img, true_lm, 6, (0, 0, 255), cv2.FILLED)    

    def show_state(self):
        img = np.zeros((1000,1000,3), dtype=np.uint8)
        self.draw_real(img)
        self.draw_particles(img)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))
        
    # this is for micro simulation only
    def sense(self, map):
        ldmrks = []
        fov = CAM_FOV/2*pi/180
        lims = [add_angle(self.teta, fov), add_angle(self.teta, -fov)]
        for lm in map:
            d = sqrt(pow(lm[0]-self.x, 2) + pow(lm[1]-self.y, 2))
            teta_d = atan2((lm[1]-self.y), (lm[0]-self.x))
            # add some noise
            d += np.random.normal(0, abs(0.2*d))
            teta_d += np.random.normal(0, abs(0.2*teta_d))
            if d <= 2: # sense only if its close
                ldmrks.append([d, teta_d])  
        return ldmrks

    # check if a ldmark detected was already found, AKA perform very basic data association
    def ldmrk_was_found(self, ldmrk):
        for lm in self.Xt[0].ldmrks:
            if point_is_close(lm, ldmrk):
                return 1
        return 0


    def callback(self, odom, imu):
        # compute shift in particle due to control, Xt = Xt-1 + Ut + Et
        if not(self.prev_t): # if there is no previous time
            self.prev_t = odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000
            return
        if odom.twist.twist.angular.z < 0.00001 and odom.twist.twist.linear.x < 0.00001:
            return
        curr_t = odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000
        timing = (curr_t - self.prev_t)/1000000000
        self.prev_t = curr_t
        delta_teta = odom.twist.twist.angular.z * timing
        delta_x = odom.twist.twist.linear.x * timing
        
        #calculate robot position (for micro-penis)
        self.teta = add_angle(self.teta, delta_teta + np.random.normal(0, abs(delta_teta/(2*1.645))))
        self.x += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*cos(self.teta)
        self.y += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*sin(self.teta)

        # update particles with control input
        for i in range(len(self.Xt)):
            self.Xt[i].teta = add_angle(self.Xt[i].teta, delta_teta + np.random.normal(0, abs(delta_teta/(2*1.645))))
            self.Xt[i].x += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*cos(self.Xt[i].teta)
            self.Xt[i].y += (delta_x + np.random.normal(0, abs(delta_x/(5*1.645))))*sin(self.Xt[i].teta)
        
        # update particles based on sensor data
        found_ldmrks = self.sense(MAP)      # vector of pairs [d, teta]
        # perform data association 
        # TODO: This loop is O(N*M*M_found), i think it can be O(N*log(M))
        for p in self.Xt: # for each particle
            for lm in found_ldmrks: # for each landmark found in this measurement ( this number is a small one so its not computanionally heavy)
                for seen_lm in p.ldmrks: # the number of landmarks already seen, may be heavy depending on number of landmarks
                    if point_is_close(lm, seen_lm, 0.3): # assume the measurement corresponds to a landmark that has been found
                        seen_lm.update(lm[0], lm[1])
                        continue
                    # if landmark is not one that has been previously found, add it to the list
                    x = p.x + lm[0]*cos(p.teta + lm[1])
                    y = p.y + lm[0]*sin(p.teta + lm[1])
                    p.ldmrks.append(LandmarkEKF(np.array([x,y]), np.matrix[[0.1, 0], [0, 0.1]], 0.1, 0.1))


        # calculate weights  


        self.show_state()




def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles', Image, queue_size=2)

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