import cv2
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


# https://stackoverflow.com/questions/64200418/optimizing-a-traveling-salesman-algorithm-time-traveler-algorithm
 
# Load the image
image = cv2.imread('Cover.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0.01)

# Perform edge detection to find the contours
edges = cv2.Canny(blurred, 77, 155)

# Find all contours in the edge-detected image
all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by their area (largest to smallest)
all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

# Function to calculate the centroid of a contour
def contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        # Return a sentinel value (e.g., None) if the contour has zero area
        return None

# Extract the centroids of all valid contours
centroids = [contour_centroid(contour) for contour in all_contours if contour_centroid(contour) is not None]

# Create a distance matrix between all centroids
distance_matrix = distance_matrix(centroids, centroids)

# Solve the Traveling Salesman Problem to find the optimized order of centroids
row_ind, col_ind = linear_sum_assignment(distance_matrix)

# Rearrange the contours based on the optimized order
optimized_contours = [all_contours[i] for i in col_ind]

# Extract the centroids of the optimized contours
optimized_centroids = [contour_centroid(contour) for contour in optimized_contours if contour_centroid(contour) is not None]

# Function to check if two line segments intersect
def intersects(p1, q1, p2, q2):
    def on_segment(p, q, r):
        if (
            min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        ):
            return True
        return False

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q1, q2):
        return True
    if o3 == 0 and on_segment(p2, q2, p1):
        return True
    if o4 == 0 and on_segment(p2, q2, q1):
        return True

    return False

# Function to optimize the path while avoiding intersections
def optimize_path(centroids, optimized_order):
    path = optimized_order.copy()
    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 2, n):
                if intersects(
                    centroids[path[i - 1]],
                    centroids[path[i]],
                    centroids[path[j - 1]],
                    centroids[path[j]],
                ):
                    # Swap the order of points between i and j
                    path[i:j] = path[i:j][::-1]
                    improved = True
    return path

# Optimize the path while avoiding intersections
optimized_order = list(range(len(optimized_centroids)))
optimized_order = optimize_path(optimized_centroids, optimized_order)

# Concatenate the contours based on the optimized order
concatenated_contour = np.concatenate([optimized_contours[i] for i in optimized_order])

# Create a blank image to draw the concatenated contour
concatenated_image = np.zeros_like(image)

# Draw the concatenated contour on the image
cv2.drawContours(concatenated_image, [concatenated_contour], -1, (0, 255, 0), 2)

# Extract the x and y coordinates of the concatenated contour
x_concatenated = [point[0][0] for point in concatenated_contour]
y_concatenated = [point[0][1] for point in concatenated_contour]


# # Create a scatter plot of the concatenated contour
# plt.figure(figsize=(8, 8))
# plt.scatter(x_concatenated, y_concatenated, c='g', marker='o', s=10)
# plt.plot(x_concatenated, y_concatenated, 'g-')  # Connect the points with a line
# plt.title('Concatenated Contour (Avoiding Intersections)')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.gca().invert_yaxis()  # Invert the y-axis to match the image orientation

# # Show the plot
# plt.show(block=False)




# Extract x and y values from the sorted path
x_values = [coord[0][0] for coord in concatenated_contour]
y_values = [coord[0][1] for coord in concatenated_contour]





# Create audio data from coordinates
sample_rate = 96000  # You can adjust the sample rate
duration = 5  # You can adjust the duration

x_audio = x_values
y_audio = y_values

# Normalize audio data to the range [-1, 1]
x_audio_normalized = (2 * (x_audio - np.min(x_audio)) / (np.max(x_audio) - np.min(x_audio)) - 1) * 1
y_audio_normalized = (2 * (y_audio - np.min(y_audio)) / (np.max(y_audio) - np.min(y_audio)) - 1) * -1

# Create a WAV file with two channels (stereo) for x and y coordinates
with wave.open('output_audio.wav', 'w') as wav_file:
    wav_file.setnchannels(2)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    

    seconds_target=6
    frames=0
    while frames/sample_rate < seconds_target:
        for i in range(len(x_audio_normalized)):
            wav_file.writeframesraw(struct.pack('<hh', int(x_audio_normalized[i] * 32767), int(y_audio_normalized[i] * 32767)))
            frames += 1

# Plot the optimized path of x and y coordinates
plt.figure(figsize=(8, 6))
plt.plot(x_audio_normalized, y_audio_normalized, marker='o', linestyle='-')
plt.title('Optimized Path of X and Y Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
# plt.show(block=False)
plt.show()

print(f'Audio saved to output_audio.wav')
