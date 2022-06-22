
import cv2
import numpy as np
from random import randint
from scipy.linalg import norm
from math import atan, sin, cos


def show_image(img):
    '''
    Displays an image
    Args:
        img(a NumPy array of type uint 8) an image to be
        dsplayedl
    '''
    
    cv2.imshow('', img)
    cv2.waitKey(10000)
    
def generate_color():
    '''
    Generates a random combination
    of red, green and blue channels
    Returns:
        (r,g,b), a generated tuple
    '''
    col = []
    for i in range(3):
        col.append(randint(0, 255))
        
    return tuple(col)

def create_test_set():
    
    #create canvas on which the triangles will be visualized
    canvas = np.full([400,400], 255).astype('uint8')
    
    #convert to 3 channel RGB for fun colors!
    canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
    
    #initialize triangles as sets of vertex coordinates (x,y)
    triangles = []
    triangles.append(np.array([250,250, 250,150, 300,250]))
    #tr1 translated by 50 points on both axis
    triangles.append(triangles[0] - 50)
    #tr1 shrinked and consequently translated as well
    triangles.append((triangles[0] / 2).astype(np.int))
    #tr1 rotated by 90 defrees annd translated by 20 pixels
    triangles.append(np.array([250,250,150,250, 250, 200]) - 20)
    #a random triangle
    triangles.append(np.array([360,240, 370,100, 390, 240]))
    
    return canvas, triangles

def draw_shapes(canvas, shapes):
    '''
    Draws shapes on canvas
    Args:
        canvas(a NumPy matrix), a background on which
        shapes are drawn
        shapes(list), shapes to be drawn
    '''
    
    for sh in shapes:
        pts = sh.reshape((-1,1,2))
        color = generate_color()
        cv2.polylines(canvas, [pts], True, color, 2)
    
    show_image(canvas)

def get_translation(shape):
  '''
  Calculates a translation for x and y
  axis that centers shape around the
  origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
   Returns:
    translation([x,y]) a NumPy array with
    x and y translationcoordinates
  '''
  
  mean_x = np.mean(shape[::2]).astype(np.int)
  mean_y = np.mean(shape[1::2]).astype(np.int)
  
  return np.array([mean_x, mean_y])

def translate(shape):
  '''
  Translates shape to the origin
  Args:
    shape(2n x 1 NumPy array) an array 
    containing x coodrinates of shape
    points as first column and y coords
    as second column
  '''
  mean_x, mean_y = get_translation(shape)
  shape[::2] -= mean_x
  shape[1::2] -= mean_y

def get_rotation_scale(reference_shape, shape):
    '''
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and 
        alignment
        
        shape(2nx1 NumPy array), a shape that is scaled
        and aligned
        
    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    '''
    
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

def rotate(shape, theta):
    '''
    Rotates a shape by angle theta
    Assumes a shape is centered around 
    origin
    Args:
        shape(2nx1 NumPy array) an shape to be rotated
        theta(float) angle in radians
    Returns:
        rotated_shape(2nx1 NumPy array) a rotated shape
    '''
    
    matr = get_rotation_matrix(theta)
    
    #reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1,2)).T
    
    #rotate
    rotated_shape = np.dot(matr, temp_shape)
    
    return rotated_shape.T.reshape(-1)


def procrustes_analysis(reference_shape, shape):
    '''
    Scales, and rotates a shape optimally to
    be aligned with a reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference alignment
        
        shape(2nx1 NumPy array), a shape that is aligned
        
    Returns:
        aligned_shape(2nx1 NumPy array), an aligned shape
        translated to the location of reference shape
    '''
    #copy both shapes in caseoriginals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)
 
    translate(temp_ref)
    translate(temp_sh)

   
    #get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    
    #scale, rotate both shapes
    #temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)
    
    return aligned_shape

if __name__ == '__main__':

    canvas, triangles = create_test_set()

    ref = [ 1.11186854, -0.99656657 ,-0.9501971 , -0.00512851  ,0.15564917 ,-2.00017548,
 -0.94977429,  0.9558844 ,  2.02466692 , 2.03290528  ,1.97740825  ,3.06647447,
 -0.88556303 ,-2.05837278 , 3.15491252 ,-3.99106038 , 3.20051511, -1.01166228]

    shape =[ 1.14883702 ,-0.92321429 ,-0.89119565, -0.05043701,  0.24630644 ,-1.9960335,
 -0.94110346 , 0.91458247 , 1.98313063 , 2.06240182 , 1.9039985 ,  3.07140482,
 -0.7645348  ,-2.10486064 , 3.36085389 ,-3.73181891 , 3.17604068 ,-0.75718961]

    x,y = get_translation(ref)
    #create array for new shapes, append reference shape to it
    new_shapes = []
    new_shapes.append(ref)

    
    # superimpose all shapes to reference shape
    new_shape = procrustes_analysis(ref, shape)
    new_shape[::2] = new_shape[::2] + x
    new_shape[1::2] = new_shape[1::2] + y
    new_shapes.append(new_shape)
    new_shapes = np.array(new_shapes)

    new_shapes = new_shapes*50 +200
    draw_shapes(canvas, new_shapes.astype(int))

# if __name__ == '__main__':
    
#     canvas, triangles = create_test_set()
#     draw_shapes(canvas, triangles)
    
