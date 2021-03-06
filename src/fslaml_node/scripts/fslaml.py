#!/usr/bin/env python3

from mmap import MAP_ANON
import time
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
from fiducial_msgs.msg import FiducialTransformArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from proscrutes import *


M_PARTICLES = 100       # number os particles 
KIDNAP_TRESH = 0.001     # minimum sum of weights that are acceptable during normal excution of the algorithm

QT = np.diag([0.2, np.deg2rad(20)])         # sensor model covariance
#R = np.diag([0.25, np.deg2rad(15)])        # motion model covariance

# for micro simulation
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


# Defines the shape of the particles
#TODO: add garbage collection?
class Particle():
    def __init__(self, x, y, teta, ldmrks):
        self.x = x
        self.y = y
        self.teta = teta
        self.ldmrks = ldmrks
        self.trajectory = []

    def copy(self):
        new = Particle(self.x, self.y, self.teta, self.ldmrks)
        return new


class LandmarkEKF():
    def __init__(self, mean, sigma, id) -> None:
        self.mean = np.array(np.reshape(mean, (2,1)))       # [mean_x, mean_y]
        self.sigma = np.array(np.reshape(sigma, (2,2)))     # covariance matrix
        self.id = id

    def comp_w8_gains(self, particle, z):

        # measurement prediction
        z_pred = predict_measurement(particle, self.mean)
        z = np.array([z[0], z[1]]).reshape(2,1)

        # compute jacobian of sensor model 
        H = jacobian(particle, self.mean)

        # measurement covariance
        Q = H @ self.sigma @ H.T + QT
        Q_inv = np.linalg.inv(Q)

        # compute kalman gain
        K = self.sigma @ H.T @ Q_inv
        c = (z - z_pred)
        c[1,0] = pi_2_pi(c[1,0])
        
        # Compute weight
        e = c.T @ Q_inv @ c
        det = abs(np.linalg.det(Q))
        weight = (1/(2*pi*sqrt(det)))*np.exp(-0.5*e[0,0])

        # save information for updating EKF later
        self.K = K
        self.c = c
        self.H = H
        
        return weight   

    def update(self):
        
        K = self.K
        c = self.c
        H = self.H

        # update mean: ??(t) = ??(t-1) + K (z - ???)
        self.mean = self.mean + K @ c
        
        # update covariance: ??(t) = (I - K H) ??(t-1) 
        self.sigma = (np.identity(2) - K @ H) @ self.sigma
    
    #TODO: check if this is even being usec
    def copy(self):
        new = LandmarkEKF(self.mean, self.sigma, self.id)
        return new


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi
    
def jacobian(particle, last_mean):
    Ht = np.array( 
            [[(last_mean[0,0]- particle.x)/sqrt((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2), 
              (last_mean[1,0]- particle.y)/sqrt((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2)],
            [ -(last_mean[1,0]- particle.y)/((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2),
            (last_mean[0,0]- particle.x)/((last_mean[0,0]- particle.x)**2 + (last_mean[1,0]- particle.y)**2)]])
    return Ht

def new_ldmrk(particle, z):
    mean_t = np.array([particle.x + z[0]*cos(pi_2_pi(particle.teta + z[1])), particle.y + z[0]*sin(pi_2_pi(particle.teta + z[1]))]).reshape(2,1)
    H = jacobian(particle, mean_t)
    H_inv = np.linalg.inv(H)
    sigma = H_inv @ QT @ H_inv.T
    landmark = LandmarkEKF(mean_t, sigma, z[2])
    return landmark

def predict_measurement(particle, mean):
    d = sqrt( (mean[0,0] - particle.x)**2 + (mean[1,0] - particle.y)**2 )
    teta = pi_2_pi(atan2(mean[1,0] - particle.y, mean[0,0] - particle.x) - particle.teta)
    return np.array([d, teta]).reshape(2,1)
    
# particle.ldmrks is an EFK, new_lm is a [d, teta] pair
def data_association(particle, z):
    #known data association
    for i, lm in enumerate(particle.ldmrks):
        if lm.id == z[2]:
            return (i, 100)
    return (-1, -1)

    # most likely data association 
    # TODO: change sigma matrix to be one always big enough to acomodate bad measurements
    # if len(particle.ldmrks) == 0:
    #     return (-1, -1)
    # x = particle.x + z[0]*cos(particle.teta + z[1])
    # y = particle.y + z[0]*sin(particle.teta + z[1])
    # max, max_i = (0,0)
    # for i, lm in enumerate(particle.ldmrks):
    #     temp = np.array([[x-lm.mean[0,0]], [y-lm.mean[1,0]]])
    #     temp = temp.T @ np.linalg.inv(lm.sigma) @ temp
    #     det = np.linalg.det(lm.sigma)
    #     p = (1/(2*pi*sqrt(abs(det)))) * np.exp(-0.5*temp[0,0])
    #     if p > max:
    #         max = p
    #         max_i = i
    # return (max_i, max)


# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, map_pub, image_pub) -> None:
        self.pub = info_pub             # for debug purposes
        self.img_pub = image_pub        # for display purposes
        self.map_pub = map_pub
        self.bridge = CvBridge()
        self.iter = 0
        self.sample_counter = 0
        self.seq = 0
        self.counter = 0
        self.scatter_counter = 0
        self.mode = "SLAM"              # "SLAM" or "LOCA" if robot is in slam mode or localization mode
        self.best_map = []
        self.time = 0
        self.odom_traject = []

        # variables for saving latest ROS msgs
        self.odom_data = [0,0,0]                # latest odometry msgs
        self.prev = [0,0,0]   
        self.sensor_data = []                   # last sensor input

        # Particles   
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(Particle(0, 0, 0, []))
        self.w = np.ones(M_PARTICLES) / M_PARTICLES

        #for micro simulation (true pose of robot)
        self.x = 0
        self.y = 0
        self.teta = 0

            
    # For micro simulation
    def sense(self, map):
        detections = []
        fov = CAM_FOV/2*pi/180
        #lims = [add_angle(self.teta, fov), add_angle(self.teta, -fov)]
        for id, lm in enumerate(map):
            d = sqrt((lm[0]-self.x)**2 + (lm[1]-self.y)**2)
            teta_d = atan2((lm[1]-self.y), (lm[0]-self.x)) - self.teta
            if d <= 2: # sense only if its close
                # add some noise
                d += np.random.normal(0, abs(0.1*d))
                teta_d += np.random.normal(0, abs(0.1*teta_d))
                detections.append([d, pi_2_pi(teta_d), id])
        detections = np.array(detections)
        return detections


    # def scatter_particles(self, pose=0):
    #     if pose:
    #         v = np.random.uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
    #         self.x = 2
    #         self.y = -2
    #         self.teta = 1.6
        
    #     map = self.Xt[np.argmax(self.w)].ldmrks
    
    #     for i in range(len(self.Xt)):
    #         v = np.random.uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
    #         self.Xt[i].x = v[0]
    #         self.Xt[i].y = v[1]
    #         self.Xt[i].teta = v[2]
    #         ldmrks = []
    #         for lm in map:
    #             new_lm = lm.copy()
    #             ldmrks.append(new_lm)
    #         self.Xt[i].ldmrks = ldmrks

    def reset(self):
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(Particle(0, 0, 0, []))
        self.w = np.ones(M_PARTICLES) / M_PARTICLES


    def check_map_quality(self):
        max = np.argmax(self.w)
        av = np.zeros((2,2))
        for lm in self.Xt[max].ldmrks:
            av = av + lm.sigma
        av = av / len(self.Xt[max].ldmrks)
        print(av)


    def normalize_weights(self):        # O(M)
        sum = np.sum(self.w)
        if np.isinf(sum):
            self.w = np.ones(M_PARTICLES) / M_PARTICLES
        else:    
            self.w = np.array(self.w) / np.sum(self.w)

    
    def low_variance_resample(self):    # O(M*log(M))
        
        n_eff = 1/(sum(self.w ** 2))
        if n_eff > M_PARTICLES/2:
            return self.Xt
        Xt = []
        r = np.random.uniform(0, 1/M_PARTICLES)
        c = self.w[0]
        i = 0
        for m in range(len(self.Xt)):
            U = r + m/M_PARTICLES
            while U >= c:
                i+=1
                c = c + self.w[i]
            particle = self.Xt[i].copy()
            Xt.append(particle)

        return Xt


    def pub_map_w_id(self):
        id = np.argmax(self.w)
        p = self.Xt[id]
        h = Header(self.counter, self.time, str(id))
        poses = []
        # first pose is the robot's pose
        point = Point(p.x, p.y, p.teta)
        quat = quaternion_from_euler(0 ,0, pi_2_pi(p.teta))
        quat = Quaternion(quat[0], quat[1], quat[2], quat[3])
        pose = Pose(point, quat)
        poses.append(pose)
        
        # next poses in the array represent the landmarks
        for lm in p.ldmrks:
            point = Point(lm.mean[0,0], lm.mean[1,0], 0)
            quat = Quaternion(lm.id, lm.id, lm.id, lm.id)
            pose = Pose(point, quat)
            poses.append(pose)
        pa = PoseArray(h, poses)
        self.map_pub.publish(pa)
    

    def statistics(self):
        index = np.argmax(self.w)
        s1 = 0
        s2 = 0
        cnt = 0
        for lm in self.Xt[index].ldmrks:
            print(lm.sigma)
            s1 += lm.sigma[0,0]
            s2 += lm.sigma[1,1]
            cnt += 1
        s1 = s1/cnt
        s2 = s2/cnt
        s = (s1+s2)/2
        plt.plot(self.iter, s, 'o')
        plt.draw()        
        plt.pause(0.00000000001)


    def draw_real(self, img): #TODO add orientation to this representation so it looks nicer
        true_pos = (-1*floor(self.y*100) + 500, -1*floor(self.x*100) +500)
        arrow = (self.y + sin(self.teta)*0.3, self.x + cos(self.teta)*0.3)
        arrow = (-1*floor(arrow[0]*100) + 500, -1*floor(arrow[1]*100) +500)
        cv2.circle(img, true_pos, 4, (0, 255, 0), cv2.FILLED)
        cv2.line(img, true_pos, arrow, (255, 0, 0), 3)
        for lm in MAP:
            true_lm = (-1*floor(lm[1]*100) + 500, -1*floor(lm[0]*100) +500)
            cv2.circle(img, true_lm, 6, (0, 0, 255), cv2.FILLED)

    def draw_best(self, img):
        max = np.argmax(self.w)
        best_pos = (-1*floor(self.Xt[max].y*100) + 500, -1*floor(self.Xt[max].x*100) +500)
        p = self.Xt[max]
        arrow = (p.y + sin(p.teta)*0.3, p.x + cos(p.teta)*0.3)
        arrow = (-1*floor(arrow[0]*100) + 500, -1*floor(arrow[1]*100) +500)
        cv2.circle(img, best_pos, 4, (255, 255, 0), cv2.FILLED)
        cv2.line(img, best_pos, arrow, (255, 0, 0), 3)
        for lm in self.Xt[max].ldmrks:
            true_lm = (-1*floor(lm.mean[1,0]*100) + 500, -1*floor(lm.mean[0,0]*100) +500)
            cv2.circle(img, true_lm, 6, (255, 0, 255), cv2.FILLED)

        for z in self.sensor_data:
            p = self.Xt[max]
            x = p.x + z[0]*cos(pi_2_pi(p.teta + z[1]))
            y = p.y + z[0]*sin(pi_2_pi(p.teta + z[1]))
            end = (-1*floor(y*100) + 500, -1*floor(x*100) +500)
            cv2.line(img, best_pos, end, (255, 255, 255), 1)   


    def show_state(self):
        img = np.zeros((1000,1000,3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[0], img.shape[1]), (100, 50, 255), 2)
        #self.draw_real(img)
        #self.draw_particles(img)
        self.draw_best(img)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))     

    def plot_w_trajectory(self):
        plt.clf()
        max = np.argmax(self.w)
        p = self.Xt[max]
        tra = np.array(p.trajectory)
        print(tra)
        x = tra[:,0]
        y = tra[:,1]
        plt.plot(-1*y, x, 'bx')
        tra = np.array(self.odom_traject)
        x = tra[:,0]
        y = tra[:,1]
        plt.plot(-1*y, x, 'rx')
        plt.axis([-0.2,0.2, -4,4])
        plt.draw()
        plt.pause(0.00000000001)


    # save information from ROS msgs into class variables
    def callback(self, odom, aruco):
        self.odom_data = np.array([odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z])
        
        # self.sensor_data = np.array(self.sense(MAP))
        sensor_data = []
        for atf in aruco.transforms:
            x = atf.transform.translation.x
            z = atf.transform.translation.z
            id  = atf.fiducial_id
            if id == 19:
                id = 1
            d = sqrt(x**2 + z**2)
            teta = atan2(-1*x, z)
            if d < 4 and d > 0.5:
                sensor_data.append([d, teta, id])
        self.sensor_data = np.array(sensor_data)

    def process(self):
        # copy msgs info into local variables 
        odom_data = np.copy(self.odom_data)
        sensor_data = np.copy(self.sensor_data)      

        self.time = rospy.Time.now()
        self.iter +=1

        if self.prev[0] == 0:             # first message must be ignored in order to compute ??T
            self.prev = odom_data
            return

        if self.prev[0] == self.odom_data[0]:       # dont process if there are no new msgs
            return

        if abs(odom_data[1]) < 0.007 and abs(odom_data[2]) < 0.007:         # ignore messages with very little velocities (aka only noise)
            self.prev = odom_data
            return

        dT = (odom_data[0] - self.prev[0])/1000000000
        dx = self.prev[1] * dT          # use the average between self.prev and odom_data?
        dteta = self.prev[2] * dT
        self.prev = odom_data

        #calculate true robot position (for micro-simulation)
        self.x += dx*cos(self.teta)
        self.y += dx*sin(self.teta)
        self.teta = pi_2_pi(self.teta + dteta)

        # update particles with input:
        for i in range(len(self.Xt)):
            self.Xt[i].x += (dx + np.random.normal(0, abs(dx/(2*1.645))))*cos(self.Xt[i].teta)
            self.Xt[i].y += (dx + np.random.normal(0, abs(dx/(2*1.645))))*sin(self.Xt[i].teta)
            self.Xt[i].teta += (dteta + np.random.normal(0, abs(dteta/2)))
            self.Xt[i].teta = pi_2_pi(self.Xt[i].teta)

            self.Xt[i].trajectory.append([self.Xt[i].x, self.Xt[i].y])
        self.odom_traject.append([self.x, self.y])

        # update particles based on sensor data
        if len(sensor_data) == 0:        # dont update EKFs if no landmarks were found
            self.show_state()
            self.pub_map_w_id()
            self.plot_w_trajectory()
            return

        # SENSOR UPDATE
        weights = []
        ldmrks_to_update = []
        for i in range(len(self.Xt)):
            weight = 1
            for z in sensor_data:
                max_i, p = data_association(self.Xt[i], z)
                if p < 0.1 or max_i == -1:
                    # add new landmark
                    landmark = new_ldmrk(self.Xt[i], z)
                    self.Xt[i].ldmrks.append(landmark)
                else:
                    # update an already found landmark
                    w = self.Xt[i].ldmrks[max_i].comp_w8_gains(self.Xt[i], z)
                    # add the pointer to the landmark to a list so it can be updated later
                    ldmrks_to_update.append(self.Xt[i].ldmrks[max_i])
                    weight = weight * w

            weights.append(weight)
        self.w = np.array(weights)
        sumw = np.sum(self.w)

        # check if weights are OK
        if sumw < KIDNAP_TRESH:
            self.reset()
            return

        # weights are ok, update landmarks
        for lm in ldmrks_to_update:
            lm.update()

        self.show_state()
        self.pub_map_w_id()
        #self.statistics()
        self.plot_w_trajectory()

        self.normalize_weights()
        # # RESAMPLING
        self.Xt = self.low_variance_resample()



def main(args):

    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Initializing FastSLAM1.0 node withPython version: ' + sys.version)

    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    aruco_sub = Subscriber('fiducial_transforms', FiducialTransformArray)
    #scan_sub = Subscriber('lidar', PointCloud2)
    
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles_img', Image, queue_size=2)
    map_pub = rospy.Publisher('fast_slam_map', PoseArray, queue_size=2)
    # map_pub = ... TODO: inventar um mapa

    pf = ParticleFilter(info_pub, map_pub, image_pub)
    rate = rospy.Rate(5)
    ats = ApproximateTimeSynchronizer([odom_sub, aruco_sub], queue_size=200, slop=0.01, allow_headerless=False)
    ats.registerCallback(pf.callback)

    while not rospy.is_shutdown():
        pf.process()
        rate.sleep()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass