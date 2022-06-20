import numpy as np
from math import cos, sin, atan
from scipy.linalg import norm


def get_translation(shape):
  mean_x = np.mean(shape[::2]).astype(np.int)
  mean_y = np.mean(shape[1::2]).astype(np.int)
  return np.array([mean_x, mean_y])

def translate_to_zero(shape):
  mean_x, mean_y = get_translation(shape)
  shape[::2] -= mean_x
  shape[1::2] -= mean_y

# isto é que faz a magia
def get_rotation_scale(reference_shape, shape):
    #rospy.loginfo(reference_shape)
    #rospy.loginfo(shape)
    a = np.dot(shape, reference_shape) / norm(reference_shape)**2
    
    #separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]
    
    x = shape[::2]
    y = shape[1::2]
    
    b = np.sum(x*ref_y - ref_x*y) / norm(reference_shape)**2
    
    scale = np.sqrt(a**2+b**2)
    theta = atan(b / max(a, 10**-10)) #avoid dividing by 0
    
    return round(scale,1), round(theta,2)

def get_rotation_matrix(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def scale(shape, scale):
    return shape / scale

def rotate(shape, theta, pose):
    matr = get_rotation_matrix(theta)
    #reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1,2)).T
    temp_pose = pose[0:2].reshape((-1,2)).T
    #rotate
    rotated_shape = np.dot(matr, temp_shape)
    rotated_pose = np.dot(matr, temp_shape)
    return rotated_shape.T.reshape(-1), rotated_pose.T.reshape(-1)

def procrustes_analysis(reference_shape, shape, pose):
    #copy both shapes in case originals are needed later
    temp_ref = np.copy(reference_shape).reshape(-1)
    temp_sh = np.copy(shape).reshape(-1)
    translate_to_zero(temp_ref)
    translate_to_zero(temp_sh)
    translate_to_zero(pose)
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    #scale, rotate both shapes
    #temp_sh = temp_sh / scale
    aligned_shape, aligned_pose = rotate(temp_sh, theta, pose)
    
    return aligned_shape, aligned_pose