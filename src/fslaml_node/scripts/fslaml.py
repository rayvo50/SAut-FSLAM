#!/usr/bin/env python3

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
from tf.transformations import quaternion_from_euler

from proscrutes import *

M_PARTICLES = 50
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


QT = np.diag([0.3, np.deg2rad(20)])
#R = np.diag([0.25, np.deg2rad(15)]) 

# Defines the shape of the particles
#TODO: add garbage collection?
class Particle():
    def __init__(self, x, y, teta, ldmrks):
        self.x = x
        self.y = y
        self.teta = teta
        self.ldmrks = ldmrks

    def copy(self):
        new = Particle(self.x, self.y, self.teta, self.ldmrks)
        return new

# Utility for manipulating particle sets 
# TODO: this class is useless, can be removed and this becomes just a function, also, add it to other file like "particle.py"
class Particle_set():
    def __init__(self) -> None:
        pass
    
    def gen_random(self):
        #p = np.random.uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
        p = np.zeros(3)     # assume boot satarting location as inertial origin
        ldmrks = []
        return Particle(p[0], p[1], p[2], ldmrks)


class LandmarkEKF():
    def __init__(self, mean, sigma, id) -> None:
        self.mean = np.array(np.reshape(mean, (2,1)))       # [mean_x, mean_y]
        self.sigma = np.array(np.reshape(sigma, (2,2)))     # covariance matrix
        self.id = id

    def update(self, particle, z):
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
        # update mean: µ(t) = µ(t-1) + K (z - ẑ)
        self.mean = self.mean + K @ c
        # update covariance: Σ(t) = (I - K H) Σ(t-1) 
        self.sigma = (np.identity(2) - K @ H) @ self.sigma
        # if self.sigma[0,0] < 0.001 or self.sigma[1,1] < 0.001:
        #     self.sigma = np.array([[0.001, 0.00001],[0.00001,0.001]])
        # compute weight:
        e = c.T @ Q_inv @ c
        det = abs(np.linalg.det(Q))
        return (1/(2*pi*sqrt(det)))*np.exp(-0.5*e[0,0])

def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

# def normalize_ang(angle, turns=0):
#     angle += pi
#     voltas = angle//(2*pi)
#     angle = angle%(2*pi) - pi
#     if turns:
#         return (angle, voltas)
#     return angle
    
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

def draw_m_2_px(img, map, pose):
    pose = (-1*floor(pose[1]*100) + 500, -1*floor(pose[0]*100) +500)
    cv2.circle(img, pose, 1, (200,170,0), cv2.FILLED)
    map = map.reshape(-1, 2)
    for lm in map:
        lm_center = (-1*floor(lm[1]*100)+500, -1*floor(lm[0]*100)+500)
        cv2.circle(img, lm_center, 1, (0, 200, 255), cv2.FILLED)


# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, image_pub, particle_pub) -> None:
        self.pub = info_pub             # for debug purposes
        self.img_pub = image_pub        # for display purposes
        self.particle_pub = particle_pub
        self.bridge = CvBridge()
        self.p_set = Particle_set()     # object to generate particles
        self.sample_counter = 0
        self.seq = 0
        self.counter = 0

        # variables for saving latest ROS msgs
        self.prev = [0,0,0]          
        self.odom_data = [0,0,0]                # latest odometry msgs
        self.sensor_data = []                   # last sensor input

        # Particles
        self.Xt = []
        for i in range(M_PARTICLES):
            self.Xt.append(self.p_set.gen_random())
        self.w = np.ones(M_PARTICLES)  # + 1/M_PARTICLES

        #for simulation (true pose of robot)
        self.x = 0
        self.y = 0
        self.teta = 0

    def scatter(self):
        v = np.random.uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
        self.x = v[0]
        self.y = v[1]
        self.teta = v[2]
        
        for i in range(len(self.Xt)):
            v = np.random.uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
            self.Xt[i].x = v[0]
            self.Xt[i].y = v[1]
            self.Xt[i].teta = v[2]

    def align_ldmrks_and_plot(self):
        # transform data into a cute little way
        maps = []
        poses = []
        for p in self.Xt:
            map = []
            for lm in p.ldmrks:
                map.append([lm.mean[0,0], lm.mean[1,0]])
            map = np.array(map).reshape(1, 2*len(map))
            maps.append(map)
            poses.append(np.array([p.x, p.y, p.teta]))

        # for now reference will be the map from the first particle
        ref = maps[0]
        #get translation of reference landmark TODO: change this to align with first landmark
        ref_x, ref_y = get_translation(ref) 
        #create array for new shapes, append reference shape to it
        aligned_maps = []
        aligned_poses = []
        aligned_maps.append(maps[0])
        aligned_poses.append(poses[0])

        # superimpose all shapes to reference shape
        for i in range(1, M_PARTICLES):
            # still need to update poses here
            new_shape, new_pose = procrustes_analysis(ref, maps[i], poses[i])
            #new_shape, new_pose = maps[i], poses[i]
            #new_shape[::2] = new_shape[::2] + ref_x
            #new_shape[1::2] = new_shape[1::2] + ref_y
            #new_pose[0] = new_pose[0] + ref_x
            #new_pose[1] = new_pose[1] + ref_x
            aligned_maps.append(new_shape)
            aligned_poses.append(new_pose)

        aligned_maps = np.array(aligned_maps)
        aligned_poses = np.array(aligned_poses)
        
        img = np.zeros((1000,1000,3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[0], img.shape[1]), (100, 50, 255), 2)
        self.draw_real(img)
        self.draw_best(img)
        self.draw_particles
        # for i in range(1, M_PARTICLES):
        #     draw_m_2_px(img, aligned_maps[i], aligned_poses[i])
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))


    ################################################################################################################
        #TODO: this is an image representation, some kind of plot would be better, força malucos
    def draw_particles(self, img): 
        for p in self.Xt:
            pose = (-1*floor(p.y*100) + 500, -1*floor(p.x*100) +500)
            cv2.circle(img, pose, 1, (200,170,0), cv2.FILLED)
            for lm in p.ldmrks:
                #lm_center = (lm.mean[1,0] - (p.y - self.y) , lm.mean[0,0] - (p.x - self.x) )
                #lm_center = (-1*floor(lm_center[0]*100)+500, -1*floor(lm_center[1]*100)+500)
                lm_center = (-1*floor(lm.mean[1][0]*100)+500, -1*floor(lm.mean[0][0]*100)+500)
                cv2.circle(img, lm_center, 1, (0, 200, 255), cv2.FILLED)

    # show robot state in an 1000X1000 image, each 100px corresponds to 1 metre
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
        self.draw_real(img)
        #self.draw_particles(img)
        self.draw_best(img)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))

    def plot(self):
        plt.clf()
        p = self.Xt[4]
        i = 0
        for p in self.Xt:           
            plt.plot(-p.y, p.x, 'o', color=COLORS[i])
            xl = []
            yl = []
            for lm in p.ldmrks:
                xl.append(lm.mean[0,0])
                yl.append(-lm.mean[1,0])
            plt.plot(yl, xl, 'x', color=COLORS[i])
            i+=1
        plt.axis([-10,10,-10,10])
        plt.draw()        
        plt.pause(0.00000000001)

    def print_info(self):
        print("all weights")
        print(self.w)
        print("best particle")
        max = np.argmax(self.w)
        print(self.w[max])
        p = self.Xt[max]
        for lm in p.ldmrks:
            print(lm.sigma)

    def pub_info(self):
        poses = []
        h = Header(self.seq, rospy.Time.now(), "base_footprint")
        for p in self.Xt:
            point = Point(p.x, p.y, 0)
            quat = quaternion_from_euler(0 ,0, pi_2_pi(p.teta))
            quat = Quaternion(quat[0], quat[1], quat[2], quat[3])
            pose = Pose(point, quat)
            poses.append(pose)
        pa = PoseArray(h, poses)
        self.particle_pub.publish(pa)

    ###############################################################################################################
        
    # this is for micro simulation only
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

    def normalize_weights(self):        # O(M)
        sum = np.sum(self.w)
        if np.isinf(sum):
            self.w = np.ones(M_PARTICLES) / M_PARTICLES
        else:    
            self.w = np.array(self.w) / np.sum(self.w)
        # if np.sum(self.w) > 200:
        #     self.w = np.ones(M_PARTICLES) / M_PARTICLES
        #     return
        # try:
        #     self.w = np.array(self.w) / np.sum(self.w)
        # except Exception as e:
        #     self.w = np.ones(M_PARTICLES) / M_PARTICLES  
    
    def low_variance_resample(self):    # O(M*log(M))
        #TODO: maybe understand & add n_eff
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
        #self.w = np.ones(M_PARTICLES) #/M_PARTICLES  
        return Xt

    # save information from ROS msgs into class variables
    def callback(self, odom, aruco):
        self.odom_data = np.array([odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z])
        self.sensor_data = np.array(self.sense(MAP))
        #self.sensor_data = np.array([aruco.d, aruco.teta])

    def process(self):
        # copy msgs info into local vari 
        odom_data = np.copy(self.odom_data)
        sensor_data = self.sensor_data

        if self.prev[0] == 0:             # first message must be ignored in order to compute ΔT
            self.prev = odom_data
            return

        if self.prev[0] == self.odom_data[0]:       # dont process if there are no new msgs
            return

        if abs(odom_data[1]) < 0.007 and abs(odom_data[2]) < 0.007:         # ignore messages with very little velocities (aka only noise)
            self.prev = odom_data
            return

        # print(self.counter)
        # if self.counter == 1000:
        #     print("OH SHIT  I'VE BEEN KIDNAPPED TIME TO SCATTER THEESE BITCHES")
        #     self.scatter()
        # self.counter +=1

        dT = (odom_data[0] - self.prev[0])/1000000000
        dx = self.prev[1] * dT          # use the average between self.prev and odom_data?
        dteta = self.prev[2] * dT
        self.prev = odom_data

        #calculate true robot position (for micro-simulation)
        self.x += dx*cos(self.teta)
        self.y += dx*sin(self.teta)
        self.teta = pi_2_pi(self.teta + dteta)
        for i in range(len(self.Xt)):
            self.Xt[i].x += (dx + np.random.normal(0, abs(dx/(1*1.645))))*cos(self.Xt[i].teta)
            self.Xt[i].y += (dx + np.random.normal(0, abs(dx/(1*1.645))))*sin(self.Xt[i].teta)
            self.Xt[i].teta += (dteta + np.random.normal(0, abs(dteta)))
            self.Xt[i].teta = pi_2_pi(self.Xt[i].teta)

        # update particles based on sensor data
        if len(sensor_data) == 0:        # dont update EKFs if no landmarks were found
            self.align_ldmrks_and_plot()
            return



        # SENSOR UPDATE
        weights = []
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
                    w = self.Xt[i].ldmrks[max_i].update(self.Xt[i], z)
                    weight = weight * w
            weights.append(weight)
        self.w = np.array(weights)

        #if np.sum(self.w) < 0.001:
            # entrar em modo localization:
                # não dar update em EKFs
                # espalhar bué as partículas
                # calcular spreadness nas particulas e ver quando é que a partir de m certro valor a localization está done
                # voltar a passar para o modo slam

        self.normalize_weights()

        self.align_ldmrks_and_plot()
        
        # RESAMPLING
        if self.sample_counter > 10:
            self.Xt = self.low_variance_resample()
            self.sample_counter = 0
        self.sample_counter +=1


def main(args):

    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Initializing FastSLAM1.0 node withPython version: ' + sys.version)

    odom_sub = Subscriber('odom', Odometry)
    aruco_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles_img', Image, queue_size=2)
    particle_pub = rospy.Publisher('particles_poses', PoseArray, queue_size=2)

    pf = ParticleFilter(info_pub, image_pub, particle_pub)
    rate = rospy.Rate(10)
    ats = ApproximateTimeSynchronizer([odom_sub, aruco_sub], queue_size=10, slop=0.3, allow_headerless=False)
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