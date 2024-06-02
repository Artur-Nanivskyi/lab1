import numpy as np
import matplotlib.pyplot as plt
import cv2


object1 = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
object2 = np.array([[0, 0], [1, 1], [0, 2], [-1, 1], [0, 0]])


def plot_objects(objects, titles):
    fig, axs = plt.subplots(1, len(objects), figsize=(15, 5))
    if len(objects) == 1:
        axs = [axs]
    for ax, obj, title in zip(axs, objects, titles):
        ax.plot(obj[:, 0], obj[:, 1], marker='o')
        ax.set_title(title)
        ax.grid()
        ax.set_aspect('equal')
    plt.show()

plot_objects([object1, object2], ['Object 1', 'Object 2'])


def rotate(obj, angle):
    rad = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    result = obj @ rotation_matrix.T
    print("Rotation Matrix:")
    print(rotation_matrix)
    return result

def scale(obj, sx, sy):
    scaling_matrix = np.array([[sx, 0], [0, sy]])
    result = obj @ scaling_matrix.T
    print("Scaling Matrix:")
    print(scaling_matrix)
    return result

def reflect(obj, axis):
    if axis == 'x':
        reflection_matrix = np.array([[1, 0], [0, -1]])
    elif axis == 'y':
        reflection_matrix = np.array([[-1, 0], [0, 1]])
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    result = obj @ reflection_matrix.T
    print("Reflection Matrix:")
    print(reflection_matrix)
    return result

def shear(obj, shx, shy):
    shearing_matrix = np.array([[1, shx], [shy, 1]])
    result = obj @ shearing_matrix.T
    print("Shearing Matrix:")
    print(shearing_matrix)
    return result

def custom_transform(obj, matrix):
    result = obj @ matrix.T
    print("Custom Transformation Matrix:")
    print(matrix)
    return result


rotated_object1 = rotate(object1, 45)
scaled_object1 = scale(object1, 2, 0.5)
reflected_object1 = reflect(object1, 'x')
sheared_object1 = shear(object1, 1, 0)
custom_transformed_object1 = custom_transform(object1, np.array([[1, 2], [3, 4]]))

plot_objects([rotated_object1, scaled_object1, reflected_object1, sheared_object1, custom_transformed_object1],
             ['Rotated Object 1', 'Scaled Object 1', 'Reflected Object 1', 'Sheared Object 1', 'Custom Transformed Object 1'])


rotated_object2 = rotate(object2, 45)
scaled_object2 = scale(object2, 2, 0.5)
reflected_object2 = reflect(object2, 'x')
sheared_object2 = shear(object2, 1, 0)
custom_transformed_object2 = custom_transform(object2, np.array([[1, 2], [3, 4]]))

plot_objects([rotated_object2, scaled_object2, reflected_object2, sheared_object2, custom_transformed_object2],
             ['Rotated Object 2', 'Scaled Object 2', 'Reflected Object 2', 'Sheared Object 2', 'Custom Transformed Object 2'])




object3D = np.array([[0, 0, 0], [1, 0.2, 0.5], [0.4, 1, -0.5], [0.5, 0.4, 0.3], [0, 0.8, -0.4],
                     [-0.5, 0.4, 0.7], [-0.4, 1, -0.2], [-1, 0.2, 0.1], [0, 0, 0]])


def plot_object_3D(obj, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], marker='o')
    ax.set_title(title)
    plt.show()


def rotate_3D(obj, angle, axis):
    rad = np.deg2rad(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    elif axis == 'y':
        rotation_matrix = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    elif axis == 'z':
        rotation_matrix = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")
    return obj @ rotation_matrix.T

def scale_3D(obj, sx, sy, sz):
    scaling_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])
    return obj @ scaling_matrix.T


rotated_object3D = rotate_3D(object3D, 45, 'x')
scaled_object3D = scale_3D(object3D, 2, 0.5, 1.5)


plot_object_3D(object3D, 'Original 3D Object')
plot_object_3D(rotated_object3D, 'Rotated 3D Object (X-axis, 45 degrees)')
plot_object_3D(scaled_object3D, 'Scaled 3D Object (2, 0.5, 1.5)')


def rotate_opencv(obj, angle):
    center = (0, 0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.transform(np.array([obj]), rotation_matrix)[0]

def scale_opencv(obj, sx, sy):
    scaling_matrix = np.array([[sx, 0, 0], [0, sy, 0]], dtype=float)
    return cv2.transform(np.array([obj]), scaling_matrix)[0]

def reflect_opencv(obj, axis):
    if axis == 'x':
        reflection_matrix = np.array([[1, 0, 0], [0, -1, 0]], dtype=float)
    elif axis == 'y':
        reflection_matrix = np.array([[-1, 0, 0], [0, 1, 0]], dtype=float)
    else:
        raise ValueError("Axis must be 'x' or 'y'")
    return cv2.transform(np.array([obj]), reflection_matrix)[0]

def shear_opencv(obj, shx, shy):
    shearing_matrix = np.array([[1, shx, 0], [shy, 1, 0]], dtype=float)
    return cv2.transform(np.array([obj]), shearing_matrix)[0]

def custom_transform_opencv(obj, matrix):
    return cv2.transform(np.array([obj]), matrix)[0]


rotated_object1_cv = rotate_opencv(object1, 45)
scaled_object1_cv = scale_opencv(object1, 2, 0.5)
reflected_object1_cv = reflect_opencv(object1, 'x')
sheared_object1_cv = shear_opencv(object1, 1, 0)
custom_transformed_object1_cv = custom_transform_opencv(object1, np.array([[1, 2, 0], [3, 4, 0]], dtype=float))

plot_objects([rotated_object1_cv, scaled_object1_cv, reflected_object1_cv, sheared_object1_cv, custom_transformed_object1_cv],
             ['Rotated Object 1 (OpenCV)', 'Scaled Object 1 (OpenCV)', 'Reflected Object 1 (OpenCV)', 'Sheared Object 1 (OpenCV)', 'Custom Transformed Object 1 (OpenCV)'])


rotated_object2_cv = rotate_opencv(object2, 45)
scaled_object2_cv = scale_opencv(object2, 2, 0.5)
reflected_object2_cv = reflect_opencv(object2, 'x')
sheared_object2_cv = shear_opencv(object2, 1, 0)
custom_transformed_object2_cv = custom_transform_opencv(object2, np.array([[1, 2, 0], [3, 4, 0]], dtype=float))

plot_objects([rotated_object2_cv, scaled_object2_cv, reflected_object2_cv, sheared_object2_cv, custom_transformed_object2_cv],
             ['Rotated Object 2 (OpenCV)', 'Scaled Object 2 (OpenCV)', 'Reflected Object 2 (OpenCV)', 'Sheared Object 2 (OpenCV)', 'Custom Transformed Object 2 (OpenCV)'])


image = cv2.imread('IMG_9965.JPG')
if image is None:
    raise ValueError("Image not found. Make sure the path is correct.")


rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), 45, 1), (image.shape[1], image.shape[0]))
scaled_image = cv2.resize(image, None, fx=2, fy=0.5)
reflected_image = cv2.flip(image, 0)
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title('Scaled Image')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB))
plt.title('Reflected Image')

plt.show()

