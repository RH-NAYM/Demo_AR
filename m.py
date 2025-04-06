import cv2
import cv2.aruco as aruco

# Set the parameters for the marker
marker_id = 23  # You can choose any ID between 0-1023
marker_size = 500  # Marker size in pixels
dict_type = aruco.DICT_4X4_50  # Choose the type of marker dictionary (4x4, 5x5, etc.)

# Create the dictionary using the correct function
aruco_dict = aruco.getPredefinedDictionary(dict_type)

# Generate the marker image
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Save the marker to a file
cv2.imwrite("aruco_marker.png", marker_image)

# Optionally display the marker
cv2.imshow("AR Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
