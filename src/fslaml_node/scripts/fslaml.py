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
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# try:
#     import matplotlib.pyplot as plt
#     SHOW = 1
# except BaseException as error:
#     pass

M_PARTICLES = 20
ROOM_SIZE = 5
MAP = np.array([
    [2,2],
    [3,-4],
    [-3, 2],
    [1,-1],
    [-1,-2],
    [2,3]
    # [3,-1],
    # [-1,0],
    # [0,-2],
    # [-1,1]
    ])
CAM_FOV = 90

Q = np.diag([0.25, np.deg2rad(15)]) 
#R = np.diag([0.25, np.deg2rad(15)]) 

# Defines the shape of the particles
class Particle():
    def __init__(self, x, y, teta, ldmrks, w):
        self.x = x
        self.y = y
        self.teta = teta
        self.ldmrks = ldmrks

# Utility for manipulating particle sets 
class Particle_set():
    def __init__(self) -> None:
        pass
    
    def gen_random(self):
        #p = uniform((-ROOM_SIZE, -ROOM_SIZE, -pi/2),(ROOM_SIZE, ROOM_SIZE, pi/2),3)
        p = np.zeros(3)     # assume boot satarting location as inertial origin
        ldmrks = []
        return Particle(p[0], p[1], p[2], ldmrks, 1/M_PARTICLES)


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
        self.w = 1                                          # default weight for landmark

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
        # update mean: µ(t) = µ(t-1) + K (z - ẑ)
        self.mean = self.mean + K @ c
        # update covariance: Σ(t) = (I - K H) Σ(t-1) 
        self.sigma = (np.identity(2) - K @ H) @ self.sigma
        # weight:
        e = c.T @ Qt_inv @ c
        det = abs(np.linalg.det(Qt))
        self.w = (1/sqrt(2*pi*det))*np.exp(-0.5*e[0,0])
    
# particle.ldmrks is an EFK, new_lm is a [d, teta] pair
def data_association(particle, new_lm):
    if len(particle.ldmrks) == 0:
        return (-1, -1)
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
        p = (1/(2*pi*sqrt(abs(det)))) * np.exp(-0.5*temp[0][0])
        if p > max:
            max = p
            max_i = i
    #rospy.loginfo("(max_i, p), calculate probability--")
    #rospy.loginfo((max_i, p))
    return (max_i, max)

# Main class for implementing ROS stuff
class ParticleFilter():
    def __init__(self, info_pub, image_pub, particle_pub) -> None:
        self.pub = info_pub             # for debug purposes
        self.img_pub = image_pub        # for display purposes
        self.particle_pub = particle_pub
        self.bridge = CvBridge()
        self.p_set = Particle_set()     # object to generate particles
        self.prev = [0,0,0]             # previous control input 
        self.sample_counter = 0
        self.counter = 0

        self.Xt = []                                # Particles
        for i in range(M_PARTICLES):
            self.Xt.append(self.p_set.gen_random())
        self.w = np.ones(M_PARTICLES)  # + 1/M_PARTICLES

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
        cv2.circle(img, best_pos, 4, (255, 255, 0), cv2.FILLED)
        for lm in self.Xt[max].ldmrks:
            true_lm = (-1*floor(lm.mean[1,0]*100) + 500, -1*floor(lm.mean[0,0]*100) +500)
            cv2.circle(img, true_lm, 6, (255, 0, 255), cv2.FILLED) 

    def show_state(self):
        img = np.zeros((1000,1000,3), dtype=np.uint8)
        cv2.rectangle(img, (0,0), (img.shape[0], img.shape[1]), (100, 50, 255), 2)
        self.draw_real(img)
        self.draw_particles(img)
        self.draw_best(img)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(img))

    def plot(self):
        if self.counter %10 == 0:
            x = []
            y = []
            for p in self.Xt:
                x.append(p.x)
                y.append(-p.y)
            plt.plot(x, y, 'o')
            plt.axis([-5,5,-5,5])
            plt.draw()        
            plt.pause(0.00000000001)
        self.counter +=1
    
    def send_info(self):
        poses = []
        h = Header(self.counter, rospy.Time.now(), "base_footprint")
        for p in self.Xt:
            point = Point()
            point.x = p.x
            point.y = p.y
            point.z = 0
            quat = quaternion_from_euler(0 ,0, pi_2_pi(p.teta))
            quat = Quaternion(quat[0], quat[1], quat[2], quat[3])
            pose = Pose()
            pose.position = point
            pose.orientation = quat
            poses.append(pose)
        pa = PoseArray(h, poses)
        self.particle_pub.publish(pa)

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
                detections.append([d, pi_2_pi(teta_d)])
        detections = np.array(detections)
        return detections

    def normalize_weights(self):        # O(M)
        w_sum = np.sum(self.w)
        return np.array(self.w) / w_sum    
    
    def low_variance_resample(self):    # O(M) ou O(M²) ?
        #TODO: Add n_eff
        Xt = []
        r = np.random.uniform(0, 1/M_PARTICLES)
        #rospy.loginfo(self.w)
        c = self.w[0]
        i = 0
        for m in range(len(self.Xt)):
            U = r + m/M_PARTICLES
            while U > c:
                i+=1
                c = c + self.w[i]
            Xt.append(self.Xt[i])
        self.w = np.ones(M_PARTICLES) #/M_PARTICLES  
        return Xt

    def callback(self, odom, imu):
        # first message must be ignored in order to compute ΔT
        if self.prev == [0,0,0]:
            self.prev = [odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z]
            return

        if odom.twist.twist.linear.x < 0.007 and odom.twist.twist.angular.z < 0.007:
            self.prev = self.prev = [odom.header.stamp.nsecs + odom.header.stamp.secs*1000000000, odom.twist.twist.linear.x, odom.twist.twist.angular.z]
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
        self.teta = pi_2_pi(self.teta)
        #rospy.loginfo((self.x, self.y, self.teta))
        for i in range(len(self.Xt)):       # TODO: find better variance values 
            self.Xt[i].x += (dx + np.random.normal(0, abs(dx/(5*1.645))))*cos(self.Xt[i].teta)
            self.Xt[i].y += (dx + np.random.normal(0, abs(dx/(5*1.645))))*sin(self.Xt[i].teta)
            self.Xt[i].teta += (dteta + np.random.normal(0, abs(dteta/(2*1.645))))
            self.Xt[i].teta = pi_2_pi(self.Xt[i].teta) 
        
        # update particles based on sensor data
        detections = self.sense(MAP)    # measurement = [d,teta] 
        detection_tresh = 0.01 # TODO: find the right value for this
        #rospy.loginfo(detections)
        if len(detections) == 0:
            self.show_state()
            self.send_info()
            return

        #print("check")

        # SENSOR UPDATE
        weights = []
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

         # calculate weights 
            w = 1
            for lm in self.Xt[i].ldmrks:
                w = w*lm.w
            weights.append(w)
        if len(weights) == M_PARTICLES:    
            self.w = np.array(weights)
        self.w = self.normalize_weights()

        #rospy.loginfo(len(self.Xt[0].ldmrks))
        self.show_state()
        self.send_info()

        # RESAMPLING
        if self.sample_counter > 10:
            self.Xt = self.low_variance_resample()
            self.sample_counter = 0
        self.sample_counter +=1

        

def main(args):
    rospy.init_node('FSLAML_node', anonymous=True)
    rospy.loginfo('Python version: ' + sys.version)
    odom_sub = Subscriber('odom', Odometry)
    imu_sub = Subscriber('imu', Imu)
    #scan_sub = Subscriber('lidar', PointCloud2)
    info_pub = rospy.Publisher('info', String, queue_size=2)
    image_pub = rospy.Publisher('particles_img', Image, queue_size=2)
    particle_pub = rospy.Publisher('particles_poses', PoseArray, queue_size=2)
    pf = ParticleFilter(info_pub, image_pub, particle_pub)

    ats = ApproximateTimeSynchronizer([odom_sub, imu_sub], queue_size=10, slop=0.3, allow_headerless=False)
    ats.registerCallback(pf.callback)
    plt.ion()
    plt.show()
    rospy.spin()


if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        #fp.close()
        pass