#!/usr/bin/env python3

from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import time
import sys
from cv2 import CAP_OPENNI_ASUS, HOGDescriptor_DESCR_FORMAT_ROW_BY_ROW, mean
import numpy as np
from math import sqrt, pi, cos, sin, atan2, floor, exp
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
MAP = np.array([[2,2], [3,-4], [-3, 2]])
CAM_FOV = 90

Q = np.diag([1.0, np.deg2rad(15)]) ** 2
R = np.diag([1.0, np.deg2rad(15)]) ** 2

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


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

def normalize_ang(angle, turns=0):
    angle += pi
    voltas = angle//(2*pi)
    angle = angle%(2*pi) - pi
    if turns:
        return (angle, voltas)
    return angle
    
def jacobian(particle, last_mean):
    Ht = np.array(
            [[(last_mean[0,0]- particle.x)/sqrt((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2), 
              (last_mean[1,0]- particle.y)/sqrt((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2)],
            [ -(last_mean[1,0]- particle.y)/((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2),
            (last_mean[0,0]- particle.x)/((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2)]])
    return Ht

def new_ldmrk(particle, z):
    # [x, y] = [d*cos(θ + θd), d*sin(θ + θd)]
    mean_t = np.array([particle.x + z[0]*cos(pi_2_pi(particle.teta + z[1])), particle.y + z[0]*sin(pi_2_pi(particle.teta + z[1]))]).reshape(2,1)
    H = jacobian(particle, mean_t)
    H_inv = np.linalg.inv(H)
    sigma = H_inv @ Q @ H_inv.T
    landmark = LandmarkEKF(mean_t, sigma)
    return landmark

def predict_measurement(particle, mean):
    d = sqrt( (mean[0,0] - particle.x)**2 + (mean[1,0] - particle.y)**2 )
    teta = atan2(mean[1,0] - particle.y, mean[0,0] - particle.x) - particle.teta
    return np.array([d, teta]).reshape(2,1)


class LandmarkEKF():
    def __init__(self, mean, sigma) -> None:
        self.mean = np.array(np.reshape(mean, (2,1)))       # [mean_x, mean_y]
        self.sigma = np.array(np.reshape(sigma, (2,2)))     # covariance matrix
        self.w = 1/M_PARTICLES

    def update(self, particle, z):
        # measurement prediction
        z_pred = predict_measurement(particle, self.mean)
        z = z.reshape(2,1)
        # compute jacobian of sensor model 
        H = jacobian(particle, self.mean)
        # measurement covariance
        Qt = H @ self.sigma @ H.T + Q
        Qt_inv = np.linalg.inv(Qt)
        # compute kalman gain
        K = self.sigma @ H.T @ Qt
        c = (z - z_pred)
        rospy.loginfo(z_pred)
        # update mean: µ(t) = µ(t-1) + K (z - ẑ)
        self.mean = self.mean + K @ c
        # update covariance: Σ(t) = (I - K H) Σ(t-1) 
        self.sigma = (np.identity(2) - K @ H) @ self.sigma
        # weight:
        e = c.T @ Qt_inv @ c
        det = abs(np.linalg.det(Qt))
        self.w = (1/sqrt(2*pi*det))*exp(-0.5*e[0,0])

        # old code:
        # ut = np.array(self.mean).reshape(2,1)
        # sigmat = np.array(self.sigma).reshape(2,2)
        # zt = np.reshape(z, (2,1))
        # # update part
        # Ht = np.array(
        #     [[(self.mean[0][0]- particle.x)/sqrt((self.mean[0][0]- particle.x)**2 + (self.mean[1][0]- particle.y)**2), 
        #       (self.mean[1][0]- particle.y)/sqrt((self.mean[0][0]- particle.x)**2 + (self.mean[1][0]- particle.y)**2)],
        #     [ -(self.mean[1][0]- particle.y)/((self.mean[0][0]- particle.x)**2 + (self.mean[1][0]- particle.y)**2),
        #     (self.mean[0][0]- particle.x)/((self.mean[0][0]- particle.x)**2 + (self.mean[1][0]- particle.y)**2)]])
        # temp = Ht @ sigmat @ Ht.T + Q
        # temp = np.reshape(temp, (2,2))
        # Kt = sigmat @ Ht.T @ np.linalg.inv(temp)
        # zt = np.reshape(z, (2,1)) 
        # # here need to check if it between 0 and 2pi, still need to make that func xd nad change all over the code for using it
        # cenas = Ht @ ut - np.array([[0], [particle.teta]])
        # rospy.loginfo("####################################")
        # rospy.loginfo(cenas)
        # rospy.loginfo("------------------------------------")
        # rospy.loginfo(zt)
        # self.mean = ut + Kt @ (zt - cenas)
        # rospy.loginfo("------------------------------------")
        # rospy.loginfo(self.mean)
        # self.sigma = (np.array([[1,0], [0,1]]) - Kt @ Ht) @ sigmat
    
# particle.ldmrks is an EFK, new_lm is a [d, teta] pair
def data_association(particle, new_lm):
    if len(particle.ldmrks) == 0:
        return (-1, 0)
    x = particle.x + new_lm[0]*cos(particle.teta + new_lm[1])
    y = particle.y + new_lm[0]*sin(particle.teta + new_lm[1])
    #rospy.loginfo("###################################")
    #rospy.loginfo("(x, y) measured--------------------")
    #rospy.loginfo((x,y))
    max, max_i = (0,0)
    #rospy.loginfo("lenght of ldmrks list--------------")
    #rospy.loginfo(len(particle.ldmrks))
    for i, lm in enumerate(particle.ldmrks):
        #rospy.loginfo("(x, y), previous EKF estimation---")
        #rospy.loginfo((lm.mean[0][0], lm.mean[1][0]))
        temp = np.array([[x-lm.mean[0][0]], [y-lm.mean[1][0]]])
        temp = temp.T @ np.linalg.inv(lm.sigma) @ temp
        det = np.linalg.det(lm.sigma)
        p = (1/(2*pi*sqrt(abs(det)))) * exp(-0.5*temp[0][0])
        if p > max:
            max = p
            max_i = i
    #rospy.loginfo("(max_i, p), calculate probability--")
    #rospy.loginfo((max_i, p))
    return (max_i, max)

# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, image_pub) -> None:
        #ros stuff
        self.pub = info_pub
        self.img_pub = image_pub
        self.bridge = CvBridge()
        # particle filter stuff
        self.p_set = Particle_set()
        self.prev = [0,0,0]
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(self.p_set.gen_random())
            self.Xt[i].w = 1/M_PARTICLES
        #for simulation (true pose of robot)
        self.x = 0
        self.y = 0
        self.teta = 0

    ################################################################################################################
        #TODO: this is an image representation, some kind of plot would be better, força malucos
    def draw_particles(self, img): 
        for p in self.Xt:
            pose = (-1*floor(p.y*100) + 500, -1*floor(p.x*100) +500)
            cv2.circle(img, pose, 1, (200,170,0), cv2.FILLED)
            for lm in p.ldmrks:
                lm_center = (-1*floor(lm.mean[1][0]*100)+500, -1*floor(lm.mean[0][0]*100)+500)
                cv2.circle(img, lm_center, 1, (0, 200, 255), cv2.FILLED)
                
    # show robot state in an 1000X1000 image, each 100px corresponds to 1 metre
    def draw_real(self, img): #TODO add orientation to this representation so it looks nicer
        true_pos = (-1*floor(self.y*100) + 500, -1*floor(self.x*100) +500)
        cv2.circle(img, true_pos, 4, (0, 255, 0), cv2.FILLED)
        for lm in MAP:
            true_lm = (-1*floor(lm[1]*100) + 500, -1*floor(lm[0]*100) +500)
            cv2.circle(img, true_lm, 6, (0, 0, 255), cv2.FILLED)    

    def show_state(self):
        img = np.zeros((1000,1000,3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[0], img.shape[1]), (100, 50, 255), 2)
        self.draw_real(img)
        self.draw_particles(img)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))
    ###############################################################################################################
        
    # this is for micro simulation only
    def sense(self, map):
        detections = []
        fov = CAM_FOV/2*pi/180
        #lims = [add_angle(self.teta, fov), add_angle(self.teta, -fov)]
        for lm in map:
            d = sqrt((lm[0]-self.x)**2 + (lm[1]-self.y)**2)
            teta_d = atan2((lm[1]-self.y), (lm[0]-self.x)) - self.teta
            # add some noise
            #d += np.random.normal(0, abs(0.2*d))
            #teta_d += np.random.normal(0, abs(0.2*teta_d))
            if d <= 2: # sense only if its close
                detections.append([d, teta_d])
        detections = np.array(detections)
        return detections

    def callback(self, odom, imu):
        # first message must be ignored in order to compute ΔT
        if self.prev == [0,0,0]:
            self.prev = [odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z]
            return

        # MOTION MODEL
        dT = (odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000 - self.prev[0])/1000000000
        dx = self.prev[1] * dT
        dteta = self.prev[2] * dT
        self.prev = self.prev = [odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z]

        #calculate robot position (for micro-simulation)
        self.x += dx*cos(self.teta)
        self.y += dx*sin(self.teta)
        self.teta += dteta
        #rospy.loginfo((self.x, self.y, self.teta))
        for i in range(len(self.Xt)):       # TODO: find better variance values 
            self.Xt[i].x += (dx + np.random.normal(0, abs(dx/(5*1.645))))*cos(self.Xt[i].teta)
            self.Xt[i].y += (dx + np.random.normal(0, abs(dx/(5*1.645))))*sin(self.Xt[i].teta)
            self.Xt[i].teta += (dteta + np.random.normal(0, abs(dteta/(2*1.645)))) 

        # update particles based on sensor data
        detections = self.sense(MAP)    # measurement = [d,teta] 
        detection_tresh = 0.01 # TODO: find the right value for this

        # SENSOR UPDATE
        for i in range(len(self.Xt)): 
            for z in detections:
                max_i, p = data_association(self.Xt[i], z)
                if p < detection_tresh or max_i == -1:  
                    # add new landmark
                    landmark = new_ldmrk(self.Xt[i], z)
                    self.Xt[i].ldmrks.append(landmark)
                else:
                    # update an already found landmark
                    self.Xt[i].ldmrks[max_i].update(self.Xt[i], z)
                
                # # old code:
                # for seen_lm in self.Xt[i].ldmrks: # the number of landmarks already seen, may be heavy depending on number of landmarks
                #     # this data association can be improved?
                #     if self.check_close(self.Xt[i], seen_lm, found_lm, 0.3): # assume the measurement corresponds to a landmark that has been found
                #         seen_lm.update(found_lm[0], found_lm[1])
                #         break
                # # if landmark is not one that has been previously found, add it to the list
                # x = self.Xt[i].x + found_lm[0]*cos(self.Xt[i].teta + found_lm[1])
                # y = self.Xt[i].y + found_lm[0]*sin(self.Xt[i].teta + found_lm[1])
                # landmark = LandmarkEKF(np.array([x,y]), np.array([[0.1, 0], [0, 0.1]]), 0.1, 0.1)
                # self.Xt[i].ldmrks.append(landmark)

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