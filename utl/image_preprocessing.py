import numpy as np

def normalize_and_window_pixel_array(pixel_array, dicom_image):
    """Normalize the pixel array to a range of 0-255 for display purposes."""
    #dicom_image is the dicom header information
    #pixel_array is the image pixels as 2D or 3D image
    image_min = 0.0
    image_max = 80.0
    
    pixel_array = pixel_array*dicom_image.RescaleSlope + dicom_image.RescaleIntercept
    
    #window the image to st head ct HU values (window centered at 40 HU, width 80 HU)
    windowed = pixel_array.copy()
    windowed[windowed < image_min] = image_min
    windowed[windowed > image_max] = image_max
    
    pixel_array = windowed.astype(np.float32)
    pixel_min = np.min(pixel_array)
    pixel_max = np.max(pixel_array)
    pixel_array = 255 * (pixel_array - pixel_min) / (pixel_max - pixel_min)
    return pixel_array.astype(np.uint8)
