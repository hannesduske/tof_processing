#!/usr/bin/env python3

import rospy
import math
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from std_msgs.msg import Header

# Default values for general parameters
DEFAULT_INPUT_TOPIC_PC2 = "sensors/tof_sensors/pcl_raw"
DEFAULT_INPUT_TOPIC_LASERSCAN = "/scan"
DEFAULT_OUTPUT_TOPIC_PRE_SLICE = "sensors/tof_sensors/pcl_pre_process"
DEFAULT_OUTPUT_TOPIC_2D_SLICE = "sensors/tof_sensors/pcl_2d_slice"
DEFAULT_OUTPUT_TOPIC_POLAR_SAMPLING = "sensors/tof_sensors/pcl_2d_polar_sampling"
DEFAULT_OUTPUT_TOPIC_DETECTED_LINES = "/sensors/tof_sensors/pcl_2d_detected_lines"
DEFAULT_OUTPUT_TOPIC_SAMPLED_LINES = "/sensors/tof_sensors/2d_sampled_lines"
DEFAULT_OUTPUT_TOPIC_POST_SLICE = "sensors/tof_sensors/pcl_post_process"
DEFAULT_OUTPUT_TOPIC_COMBINED = "sensors/tof_hybrid/hybrid_scan"
DEFAULT_PUBLISH_INTERMEDIATE_TOPICS = False

# Default values for pre processing
DEFAULT_FILTER_FLOOR = True
DEFAULT_FLOOR_BELOW_BASE_THRESHOLD_M = 0.03
DEFAULT_FILTER_HIGH_SIGMA = False
DEFAULT_HIGH_SIGMA_THRESHOLD = 6e-03

# Default values for 2D slice
DEFAULT_PROJECT_ON_PLANE = False
DEFAULT_DISTANCE_THRESHOLD = 0.05 # m
DEFAULT_PLANE_POINT = [0.0,0.0,0.10] # 5cm above base_link
DEFAULT_PLANE_NORMAL = [0.0,0.0,1.0] # x-y-Plane

# Default values for post processing
DEFAULT_REDUCTION_FILTER = True
DEFAULT_REDUCTION_FILTER_TRESHOLD_M = 0.10

DEFAULT_DETECT_LINES = True
DEFAULT_DETECT_LINES_REGRESSION = True
DEFAULT_DETECT_LINES_EPSILON = 0.1
DEFAULT_DETECT_LINES_MAX_DISTANCE_M = 0.3
DEFAULT_DETECT_LINES_MIN_POINTS = 4

DEFAULT_SAMPLE_LINES = False
DEFAULT_SAMPLE_LINES_POLAR_INCREMENT_DEG = 1.0

DEFAULT_INTERPOLATE_LASER_SCAN = False
DEFAULT_INTERPOLATE_MAX_DISTANCE_M = 0.05
DEFAULT_NTERPOLATE_LASER_ANGLE_DEG = 1.0

DEAFULT_COMBINE_LINES_AND_INTERPOLATION = False

class ToFPreProcess:
    def __init__(self):

        self.got_first_laserscan = False
        self.last_laser_msg = LaserScan()

        rospy.init_node('tof_pre_process_node', anonymous=True)
        rospy.loginfo("Starting tof pre-processing node")

        self.INPUT_TOPIC_PC2 = rospy.get_param('~INPUT_TOPIC_PC2', DEFAULT_INPUT_TOPIC_PC2)
        self.INPUT_TOPIC_LASERSCAN = rospy.get_param('~INPUT_TOPIC_LASERSCAN', DEFAULT_INPUT_TOPIC_LASERSCAN)
        self.PUBLISH_INTERMEDIATE_TOPICS = rospy.get_param('~PUBLISH_INTERMEDIATE_TOPICS', DEFAULT_PUBLISH_INTERMEDIATE_TOPICS)
        self.OUTPUT_TOPIC_PRE_SLICE = rospy.get_param('~OUTPUT_TOPIC_PRE_SLICE', DEFAULT_OUTPUT_TOPIC_PRE_SLICE)
        self.OUTPUT_TOPIC_2D_SLICE = rospy.get_param('~OUTPUT_TOPIC_2D_SLICE', DEFAULT_OUTPUT_TOPIC_2D_SLICE)
        self.OUTPUT_TOPIC_POLAR_SAMPLING = rospy.get_param('~OUTPUT_TOPIC_POLAR_SAMPLING', DEFAULT_OUTPUT_TOPIC_POLAR_SAMPLING)
        self.OUTPUT_TOPIC_DETECTED_LINES = rospy.get_param('~OUTPUT_TOPIC_DETECTED_LINES', DEFAULT_OUTPUT_TOPIC_DETECTED_LINES)
        self.OUTPUT_TOPIC_SAMPLED_LINES = rospy.get_param('~OUTPUT_TOPIC_SAMPLED_LINES', DEFAULT_OUTPUT_TOPIC_SAMPLED_LINES)
        self.OUTPUT_TOPIC_POST_SLICE = rospy.get_param('~OUTPUT_TOPIC_POST_SLICE', DEFAULT_OUTPUT_TOPIC_POST_SLICE)
        self.OUTPUT_TOPIC_COMBINED = rospy.get_param('~OUTPUT_TOPIC_COMBINED', DEFAULT_OUTPUT_TOPIC_COMBINED)

        self.FILTER_FLOOR = rospy.get_param('~FILTER_FLOOR', DEFAULT_FILTER_FLOOR)
        self.FLOOR_BELOW_BASE_THRESHOLD_M = rospy.get_param('~FLOOR_BELOW_BASE_THRESHOLD_M', DEFAULT_FLOOR_BELOW_BASE_THRESHOLD_M)
        self.FILTER_HIGH_SIGMA = rospy.get_param('~FILTER_HIGH_SIGMA', DEFAULT_FILTER_HIGH_SIGMA)
        self.HIGH_SIGMA_THRESHOLD = rospy.get_param('~HIGH_SIGMA_THRESHOLD', DEFAULT_HIGH_SIGMA_THRESHOLD)

        self.PROJECT_ON_PLANE   = rospy.get_param('~PROJECT_ON_PLANE', DEFAULT_PROJECT_ON_PLANE)
        self.DISTANCE_THRESHOLD = rospy.get_param('~DISTANCE_THRESHOLD', DEFAULT_DISTANCE_THRESHOLD)
        self.PLANE_POINT = np.array(rospy.get_param('~PLANE_POINT', DEFAULT_PLANE_POINT), dtype=float)
        self.PLANE_NORMAL = np.array(rospy.get_param('~PLANE_NORMAL', DEFAULT_PLANE_NORMAL), dtype=float)

        self.REDUCTION_FILTER                 = rospy.get_param('~REDUCTION_FILTER', DEFAULT_REDUCTION_FILTER)
        self.REDUCTION_FILTER_TRESHOLD_M      = rospy.get_param('~REDUCTION_FILTER_TRESHOLD_M', DEFAULT_REDUCTION_FILTER_TRESHOLD_M)
        self.DETECT_LINES                     = rospy.get_param('~DETECT_LINES', DEFAULT_DETECT_LINES)
        self.DETECT_LINES_REGRESSION          = rospy.get_param('~DETECT_LINES_REGRESSION', DEFAULT_DETECT_LINES_REGRESSION)
        self.DETECT_LINES_EPSILON             = rospy.get_param('~DETECT_LINES_EPSILON', DEFAULT_DETECT_LINES_EPSILON)
        self.DETECT_LINES_MAX_DISTANCE_M      = rospy.get_param('~DETECT_LINES_MAX_DISTANCE_M', DEFAULT_DETECT_LINES_MAX_DISTANCE_M)
        self.DETECT_LINES_MIN_POINTS          = rospy.get_param('~DETECT_LINES_MIN_POINTS', DEFAULT_DETECT_LINES_MIN_POINTS)
        self.SAMPLE_LINES                     = rospy.get_param('~SAMPLE_LINES', DEFAULT_SAMPLE_LINES)
        self.SAMPLE_LINES_POLAR_INCREMENT_DEG = rospy.get_param('~SAMPLE_LINES_POLAR_INCREMENT_DEG', DEFAULT_SAMPLE_LINES_POLAR_INCREMENT_DEG)
        self.INTERPOLATE_LASER_SCAN           = rospy.get_param('~INTERPOLATE_LASER_SCAN', DEFAULT_INTERPOLATE_LASER_SCAN)
        self.INTERPOLATE_MAX_DISTANCE_M       = rospy.get_param('~INTERPOLATE_MAX_DISTANCE_M', DEFAULT_INTERPOLATE_MAX_DISTANCE_M)
        self.INTERPOLATE_LASER_ANGLE_DEG      = rospy.get_param('~INTERPOLATE_LASER_ANGLE_DEG', DEFAULT_NTERPOLATE_LASER_ANGLE_DEG)
        self.COMBINE_LINES_AND_INTERPOLATION  = rospy.get_param('~COMBINE_LINES_AND_INTERPOLATION', DEAFULT_COMBINE_LINES_AND_INTERPOLATION)

        rospy.loginfo("\n\nGeneral parameters:")
        rospy.loginfo("PUBLISH_INTERMEDIATE_TOPICS: " + str(self.PUBLISH_INTERMEDIATE_TOPICS))
        rospy.loginfo("INPUT_TOPIC_PC2: " + str(self.INPUT_TOPIC_PC2))
        rospy.loginfo("INPUT_TOPIC_LASERSCAN: " + str(self.INPUT_TOPIC_LASERSCAN))
        if self.PUBLISH_INTERMEDIATE_TOPICS:
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_PRE_SLICE: " + str(self.OUTPUT_TOPIC_PRE_SLICE))
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_2D_SLICE: " + str(self.OUTPUT_TOPIC_2D_SLICE))
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_POLAR_SAMPLING: " + str(self.OUTPUT_TOPIC_POLAR_SAMPLING))
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_DETECTED_LINES: " + str(self.OUTPUT_TOPIC_DETECTED_LINES))
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_SAMPLED_LINES: " + str(self.OUTPUT_TOPIC_SAMPLED_LINES))
            rospy.loginfo("(intermediate) OUTPUT_TOPIC_POST_SLICE: " + str(self.OUTPUT_TOPIC_POST_SLICE))
        rospy.loginfo("OUTPUT_TOPIC_COMBINED: " + str(self.OUTPUT_TOPIC_COMBINED))

        rospy.loginfo("\n\nPre processing parameters:")
        rospy.loginfo("FILTER_FLOOR: " + str(self.FILTER_FLOOR))
        rospy.loginfo("FLOOR_BELOW_BASE_THRESHOLD_M: " + str(self.FLOOR_BELOW_BASE_THRESHOLD_M))
        rospy.loginfo("FILTER_HIGH_SIGMA: " + str(self.FILTER_HIGH_SIGMA))
        rospy.loginfo("HIGH_SIGMA_THRESHOLD: " + str(self.HIGH_SIGMA_THRESHOLD))

        rospy.loginfo("\n\n2D slice parameters:")
        rospy.loginfo("PROJECT_ON_PLANE : " + str(self.PROJECT_ON_PLANE))
        rospy.loginfo("DISTANCE_THRESHOLD: " + str(self.DISTANCE_THRESHOLD))
        rospy.loginfo("PLANE_POINT: " + str(self.PLANE_POINT))
        rospy.loginfo("PLANE_NORMAL: " + str(self.PLANE_NORMAL) + " (normalized)")

        rospy.loginfo("\n\nPost processing parameters:")
        rospy.loginfo("REDUCTION_FILTER: " + str(self.REDUCTION_FILTER))
        rospy.loginfo("REDUCTION_FILTER_TRESHOLD_M: " + str(self.REDUCTION_FILTER_TRESHOLD_M))
        rospy.loginfo("DETECT_LINES: " + str(self.DETECT_LINES))
        rospy.loginfo("DETECT_LINES_REGRESSION: " + str(self.DETECT_LINES_REGRESSION))
        rospy.loginfo("DETECT_LINES_EPSILON: " + str(self.DETECT_LINES_EPSILON))
        rospy.loginfo("DETECT_LINES_MAX_DISTANCE_M: " + str(self.DETECT_LINES_MAX_DISTANCE_M))
        rospy.loginfo("DETECT_LINES_MIN_POINTS: " + str(self.DETECT_LINES_MIN_POINTS))
        rospy.loginfo("SAMPLE_LINES: " + str(self.SAMPLE_LINES))
        rospy.loginfo("SAMPLE_LINES_POLAR_INCREMENT_DEG: " + str(self.SAMPLE_LINES_POLAR_INCREMENT_DEG))
        rospy.loginfo("INTERPOLATE_LASER_SCAN: " + str(self.INTERPOLATE_LASER_SCAN))
        rospy.loginfo("INTERPOLATE_MAX_DISTANCE_M: " + str(self.INTERPOLATE_MAX_DISTANCE_M))
        rospy.loginfo("INTERPOLATE_LASER_ANGLE_DEG: " + str(self.INTERPOLATE_LASER_ANGLE_DEG))
        rospy.loginfo("COMBINE_LINES_AND_INTERPOLATION: " + str(self.COMBINE_LINES_AND_INTERPOLATION))

        # Normalize plane normal to unit vector, otherwise functions in receive_pointcloud() have to be changed
        self.PLANE_NORMAL = self.PLANE_NORMAL / np.linalg.norm(self.PLANE_NORMAL)
        self.tof_sub = rospy.Subscriber(self.INPUT_TOPIC_PC2, PointCloud2, self.receive_pointcloud)
        self.laser_sub = rospy.Subscriber(self.INPUT_TOPIC_LASERSCAN, LaserScan, self.receive_laserscan)
        
        if self.PUBLISH_INTERMEDIATE_TOPICS:
            self.tof_pre_process_pub = rospy.Publisher(self.OUTPUT_TOPIC_PRE_SLICE, PointCloud2, queue_size=10)
            self.tof_2d_slice_pub = rospy.Publisher(self.OUTPUT_TOPIC_2D_SLICE, PointCloud2, queue_size=10)
            self.tof_post_process_pub = rospy.Publisher(self.OUTPUT_TOPIC_POST_SLICE, PointCloud2, queue_size=10)
            self.tof_line_pub = rospy.Publisher(self.OUTPUT_TOPIC_DETECTED_LINES, PointCloud2, queue_size= 10)
            self.tof_post_process_polar_pub = rospy.Publisher(self.OUTPUT_TOPIC_POLAR_SAMPLING, LaserScan, queue_size= 10)
            self.tof_sampled_lines_polar_pub = rospy.Publisher(self.OUTPUT_TOPIC_SAMPLED_LINES, LaserScan, queue_size= 10)
        self.tof_combined_pub = rospy.Publisher(self.OUTPUT_TOPIC_COMBINED, LaserScan, queue_size=10)

    def receive_laserscan(self, msg):
        self.got_first_laserscan = True
        self.last_laser_msg = msg

    def receive_pointcloud(self, msg):
        # Restore shape of flattened array
        msg_bytes = np.frombuffer(msg.data, dtype=np.float32)
        point_cloud = np.reshape(msg_bytes, (msg.width, 4))

        filtered_idx = np.any(point_cloud)

        ##########################################################################################
        # Filter  points that are below some threshold in the z-direction (points on on the floor)
        ##########################################################################################
        if self.FILTER_FLOOR:
            filtered_idx_floor = point_cloud[:, 2] > self.FLOOR_BELOW_BASE_THRESHOLD_M
            filtered_idx = np.multiply(filtered_idx, filtered_idx_floor)

        ##########################################################################################
        # Filter points with a high variance
        ##########################################################################################
        if self.FILTER_HIGH_SIGMA:
            filtered_idx_sigma = point_cloud[:, 3] < self.HIGH_SIGMA_THRESHOLD
            filtered_idx = np.multiply(filtered_idx, filtered_idx_sigma)

        filtered_points = point_cloud[filtered_idx, :]

        # Publist first intermediate result
        if(self.PUBLISH_INTERMEDIATE_TOPICS):    
            scan_msg_filter = self.prepare_PointCloud2_msg(filtered_points, msg.header, "sigma")
            self.tof_pre_process_pub.publish(scan_msg_filter)

        ##########################################################################################
        # Take a 2D slice from the points
        ##########################################################################################

        # Calculate the directional(!) distances of all points to the plane (Ref. Papula p.62, sec. 4.3.4)
        #distances = np.dot((points_2d[:,0:3]-self.PLANE_POINT), self.PLANE_NORMAL)/np.linalg.norm(self.PLANE_NORMAL)
        distances = np.dot((filtered_points[:,0:3]-self.PLANE_POINT), self.PLANE_NORMAL)

        # Filter points by their distance from the specified plane
        filtered_idx = np.abs(distances) < self.DISTANCE_THRESHOLD
        point_cloud_2d = filtered_points[filtered_idx]
        filtered_distances = distances[filtered_idx]

        # Project points on the 2d plane using dyadic product of distances and plane normal
        if self.PROJECT_ON_PLANE:
            #points_2d[:,0:3] = points_2d[:,0:3] - np.outer(distances, self.PLANE_NORMAL/np.linalg.norm(self.PLANE_NORMAL))
            point_cloud_2d[:,0:3] = point_cloud_2d[:,0:3] - np.outer(filtered_distances, self.PLANE_NORMAL)

        # Publist second intermediate result
        if(self.PUBLISH_INTERMEDIATE_TOPICS):
            scan_msg_filter = self.prepare_PointCloud2_msg(point_cloud_2d, msg.header, "sigma")
            self.tof_2d_slice_pub.publish(scan_msg_filter)

        ##########################################################################################
        # Post process the 2D slice
        ##########################################################################################

        point_count = np.size(point_cloud_2d, axis=0)
        if(point_count > 0):

            # Sort points by polar angle to simulate a "sorted" laserscanner
            # TODO: Only valid for 2d-slices parallel to the base-line x-y- plane !!! 
            polar_angles = np.zeros(point_count)
            polar_angles = np.arctan2(point_cloud_2d[:, 1], point_cloud_2d[:, 0]) # atan2(y/x)
            point_cloud_2d = point_cloud_2d[polar_angles.argsort()] # sort along last axis
            

            # Reduction filter
            # See "Mobile Roboter" (ISBN 978-3-642-01725-4) chapter "3.1.1 Reduktionsfilter (p.69f)" for explanation
            if self.REDUCTION_FILTER:
                
                accumulated_points = []
                reduced_point_cloud_2d = []
                start_point = point_cloud_2d[0, 0:3]

                for i in range(point_count):
                    
                    # Calculate distance between starting point and current point
                    diff = np.linalg.norm(start_point - point_cloud_2d[i, 0:3])

                    if diff > self.REDUCTION_FILTER_TRESHOLD_M:
                        reduced_point_cloud_2d.append(np.mean(np.asarray(accumulated_points), axis=0))
                        start_point = point_cloud_2d[i, 0:3]
                        accumulated_points = []
                    
                    accumulated_points.append(point_cloud_2d[i, :])

                point_cloud_2d = np.asarray(reduced_point_cloud_2d)
                point_count = np.size(point_cloud_2d, axis=0)

                # Make a LaserScan message from the sampled point
                # Have to somehow mangle the point data to discrete angles
                if self.INTERPOLATE_LASER_SCAN:
                    # Distances to origin of the points
                    polar_distances = np.linalg.norm(point_cloud_2d, axis=1)
                    # Angles of the points
                    polar_angles = np.rad2deg(np.arctan2(point_cloud_2d[:, 1], point_cloud_2d[:, 0]))
                    #self.get_logger().info(str(polar_angles))

                    point_count_polar = int(np.round(360/self.INTERPOLATE_LASER_ANGLE_DEG))
                    interpolated_distances = np.zeros(point_count_polar, dtype=np.float32)
                    intensities = np.zeros(point_count_polar, dtype=np.float32)

                    for i in range(point_count_polar):
                        angle = i * self.INTERPOLATE_LASER_ANGLE_DEG
                        if angle > 180:
                            angle -= 360

                        current_ray_vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])

                        # Get idx of two closest point to current ray
                        closest_points_idx = np.argpartition(np.abs((polar_angles - angle)), 1)[0:2]
                        p1 = point_cloud_2d[closest_points_idx[0], 0:2]
                        p2 = point_cloud_2d[closest_points_idx[1], 0:2]  
                        p1_angle = polar_angles[closest_points_idx[0]]
                        p2_angle = polar_angles[closest_points_idx[1]]                      
                        p1_dist = polar_distances[closest_points_idx[0]]
                        p2_dist = polar_distances[closest_points_idx[1]]

                        # Calculate projection from the points on the ray
                        d1 = np.dot(p1, current_ray_vector) * current_ray_vector
                        d2 = np.dot(p2, current_ray_vector) * current_ray_vector
                        #p1_proj = p1 - (p1 - d1)
                        #p2_proj = p2 - (p2 - d2)
                        
                        # Calculate the distances from the points to the ray
                        d1_dist = np.linalg.norm(p1 - d1)
                        d2_dist = np.linalg.norm(p2 - d2)

                        if np.linalg.norm(p1 - p2) < self.INTERPOLATE_MAX_DISTANCE_M:
                            # both points have to be clost to the ray, and they must lie on opposite sites of the ray
                            if (d1_dist < self.INTERPOLATE_MAX_DISTANCE_M) and (d2_dist < self.INTERPOLATE_MAX_DISTANCE_M) and (((p1_angle - angle) * (p2_angle - angle)) < 0):
                                # linear interpolation
                                interpolated_distances[i] = p1_dist + (angle - p1_angle) * ((p2_dist - p1_dist)/(p2_angle - p1_angle))
                        elif (d1_dist < d2_dist) and (d1_dist < (self.INTERPOLATE_MAX_DISTANCE_M / 6)):
                            interpolated_distances[i] = p1_dist
                        elif d2_dist < (self.INTERPOLATE_MAX_DISTANCE_M / 6): 
                            interpolated_distances[i] = p2_dist


                    interpolated_lines_msg = self.prepare_LaserScan_msg(interpolated_distances, intensities, self.INTERPOLATE_LASER_ANGLE_DEG, 0.05, 5.0, msg.header)
                    self.tof_post_process_polar_pub.publish(interpolated_lines_msg)



            # Online line detection
            # See "Mobile Roboter" (ISBN 978-3-642-01725-4) chapter "3.1.2 Linienerkennung (p.69f)" for explanation
            if self.DETECT_LINES and point_count > self.DETECT_LINES_MIN_POINTS * 2:
                #self.get_logger().info(str(np.shape(point_cloud_2d)))
                
                line_count = 0
                point_cloud_lines = np.empty((0,4), dtype=np.float32)

                start_line_point = point_cloud_2d[0, 0:3]
                prev_point = point_cloud_2d[0, 0:3]  # only for distance check, gets not saved
                prev_prev_point = np.zeros(3)
                
                current_line_idx = []
                current_line_idx.append(0)
                
                summed_up_point_distances = 0
                prev_distance_to_prev_point = 0
                
                # Iterate through the sorted points
                # Eiter a new point is added to the current line or the current line is terminated and the next potential line is evaluated
                for i in range(1, point_count):
                    
                    terminate_line = True
                    current_point = point_cloud_2d[i, 0:3]

                    # if current point is not zero in all dimensions
                    if(np.any(current_point)):
                        #self.get_logger().info("Current index: " + str(i) + ", Coords: " + str(current_point))
                        #self.get_logger().info("Start point: " + str(start_line_point) + ", Prev point: " + str(prev_point))

                        # Check if current point extends the current line or is too far off
                        distance_to_prev_point = np.linalg.norm(current_point - prev_point)
                        #self.get_logger().info("Distance to prev point: " + str(distance_to_prev_point) + ", Threshold is " + str(self.DETECT_LINES_MAX_DISTANCE_M))
                        
                        # Condition 3: The euclidean distance to the prev point may not be greater than some threshold
                        if distance_to_prev_point < self.DETECT_LINES_MAX_DISTANCE_M:
                            summed_up_point_distances += distance_to_prev_point

                            # If only start point has been saved until now, save one more point
                            #self.get_logger().info(str(len(current_line_idx)))
                            if len(current_line_idx) < 2:
                                terminate_line = False
                            else:
                                
                                distance_to_start_point = np.linalg.norm(current_point - start_line_point)
                                diff = distance_to_start_point / summed_up_point_distances
                                #self.get_logger().info("Diff to starting point: " + str(diff) + ", Threshold is " + str((1 - self.DETECT_LINES_EPSILON)))

                                # Condition 1: The direct distance from the curent point to the starting point of the line may not be much smaller than the sumed up indiviual distances between all points of the current line
                                if diff > (1 - self.DETECT_LINES_EPSILON):

                                    local_distance_1 = np.linalg.norm(current_point - prev_prev_point)
                                    local_distance_2 = prev_distance_to_prev_point + distance_to_prev_point
                                    prev_distance_to_prev_point = distance_to_prev_point

                                    local_diff = local_distance_1 / local_distance_2
                                    #self.get_logger().info("Local diff: " + str(diff) + ", Threshold is " + str((1 - self.DETECT_LINES_EPSILON)) + ", line length is: " + str(len(current_line)))
                                        
                                    # Condition 2: Same as condition 1 but seen locally around the prev point
                                    if (local_diff > (1 - self.DETECT_LINES_EPSILON)):
                                        # If all conditons are met, the index of the current point is added to the current line
                                        terminate_line = False
                                        
                    # Terminate line one of the three conditions is not met or if the last point has been reached
                    if terminate_line or i == (point_count - 1):

                        # If termination criteria is met, check if line is long enough to be saved
                        if len(current_line_idx) >= self.DETECT_LINES_MIN_POINTS:
                            current_line = np.zeros((len(current_line_idx), 4), dtype=np.float32)
                            for i in range(len(current_line_idx)):
                                current_line[i, :] = point_cloud_2d[current_line_idx[i], :]
                                current_line[i, 3] = line_count

                            point_cloud_lines = np.append(point_cloud_lines, current_line, axis=0)
                            line_count += 1

                        # Reset all variables
                        summed_up_point_distances = 0
                        prev_distance_to_prev_point = 0
                        current_line_idx = []

                        # Until discontinuity in line has been detected, one additional point will already be checked, so set starting point to previous point
                        if np.linalg.norm(current_point - point_cloud_2d[i-1, 0:3]) < self.DETECT_LINES_MAX_DISTANCE_M:
                            current_line_idx.append(i-1)
                            current_line_idx.append(i)

                            start_line_point = point_cloud_2d[i-1, 0:3]
                            prev_prev_point = point_cloud_2d[i-1, 0:3]
                            prev_point = point_cloud_2d[i, 0:3]
                        else:
                            current_line_idx.append(i)
                            start_line_point = point_cloud_2d[i, 0:3]
                            prev_prev_point = point_cloud_2d[i, 0:3]
                            prev_point = point_cloud_2d[i, 0:3]


                    else:
                        current_line_idx.append(i)
                        prev_prev_point = prev_point
                        prev_point = current_point

                line_msg = self.prepare_PointCloud2_msg(point_cloud_lines, msg.header, "line_no")
                self.tof_line_pub.publish(line_msg)

                # Linear regression to fit lines to points via PCA
                regression_lines = np.zeros((2 * line_count, 4), dtype=np.float32)
                if self.DETECT_LINES_REGRESSION:
                    for i in range(line_count):
                        current_line_points = point_cloud_lines[point_cloud_lines[:,3] == i][:, 0:3]
                        current_line_mean = np.mean(current_line_points, axis=0)
                        # Center the data
                        current_line_points = (current_line_points - current_line_mean)[:,0:2]
                        # Calculate covariance matrix
                        current_line_cov = np.cov(current_line_points, rowvar=False)
                        # Calculate eigenvalues and eigenvectors of symmetric covariance matrix
                        eig_vals, eig_vecs = np.linalg.eigh(current_line_cov)
                        # Sort eigenvectors by eigenvalue size (ascending)
                        idx = eig_vals.argsort()
                        eig_vecs = eig_vecs[idx]
                        #eig_vals = eig_vals[idx]

                        # Normalize line vector
                        current_line_vector = eig_vecs[-1, :] / np.linalg.norm(eig_vecs[-1, :])

                        # Project old end-points of line on new regression line  
                        p1 = current_line_points[0, :]
                        p2 = current_line_points[-1, :]

                        # Calculate projection from point on vector
                        d1 = np.dot(p1, current_line_vector) * current_line_vector
                        d2 = np.dot(p2, current_line_vector) * current_line_vector

                        # Subtract the difference between the projected point and the point (= projection factor) from the point
                        regression_lines[i*2 + 0, 0:2] = p1 - (p1 - d1)
                        regression_lines[i*2 + 1, 0:2] = p2 - (p2 - d2)
                        regression_lines[i*2 + 0, 0:3] += current_line_mean
                        regression_lines[i*2 + 1, 0:3] += current_line_mean
                        regression_lines[i*2 + 0, 3] = i
                        regression_lines[i*2 + 1, 3] = i

                    point_cloud_lines = regression_lines

                # Sampling the lines
                if self.DETECT_LINES and self.SAMPLE_LINES:

                    angle_increment = self.INTERPOLATE_LASER_ANGLE_DEG if (self.COMBINE_LINES_AND_INTERPOLATION) else self.SAMPLE_LINES_POLAR_INCREMENT_DEG
                    polar_angles = np.rad2deg(np.arctan2(point_cloud_lines[:, 1], point_cloud_lines[:, 0]))
                    point_count_polar = int(np.round(360/angle_increment))
                    ranges = interpolated_distances if (self.COMBINE_LINES_AND_INTERPOLATION) else np.zeros(point_count_polar, dtype=np.float32)
                    intensities = np.zeros(point_count_polar, dtype=np.float32)

                    for i in range(line_count):
                        current_line_idx = point_cloud_lines[:,3] == i
                        
                        if(np.any(current_line_idx)):
                            current_line_angles = polar_angles[current_line_idx]
                            current_line_points = point_cloud_lines[current_line_idx]

                            first_point = current_line_points[0, 0:3]
                            last_point = current_line_points[-1, 0:3]
                            line_vector = (last_point - first_point) / np.linalg.norm(last_point - first_point)

                            #increase first line-point angle to next higher discrete value
                            first_angle = min([current_line_angles[0], current_line_angles[-1]])
                            first_angle_idx = int(np.round(first_angle / angle_increment))

                            #decrease last line-point angle to next lower discrete value
                            last_angle = max([current_line_angles[0], current_line_angles[-1]])
                            last_angle_idx = int(np.round(last_angle // angle_increment))

                            point_count = last_angle_idx - first_angle_idx
                            for j in range(point_count):
                                angle = (first_angle_idx + j) * angle_increment
                                angle_idx = int(first_angle_idx + j)

                                # Calculate new points with discrete angles on the detected lines
                                # d * cos(angle) = p_x + t * line_vector_x
                                # d * sin(angle) = p_y + t * line_vector_y
                                # -> solve for d
                                nom = first_point[0] * line_vector[1] - first_point[1] * line_vector[0]
                                denom = np.cos(np.deg2rad(angle)) * line_vector[1] - np.sin(np.deg2rad(angle)) * line_vector[0]
                                distance = nom / denom

                                ranges[angle_idx] = distance
                                intensities[angle_idx] = i + 1

                    sampled_line_msg = self.prepare_LaserScan_msg(ranges, intensities, angle_increment, 0.05, 5.0, msg.header)
                    self.tof_sampled_lines_polar_pub.publish(sampled_line_msg)

            if point_count > 0:
                post_msg = self.prepare_PointCloud2_msg(point_cloud_2d, msg.header, "sigma")
                self.tof_post_process_pub.publish(post_msg)


            if(self.got_first_laserscan):
                self.tof_combined_pub.publish(self.last_laser_msg)



    def prepare_PointCloud2_msg(self, point_data, header, last_dim_name):
        # Prepare data to be published as ros message
        scan_msg = PointCloud2()

        scan_msg.header = header
        scan_msg.data = point_data.flatten().tobytes()

        scan_msg.point_step   = 4 * 4 # Dimensions per zone * bytes per dimention (float32 -> 4 bytes)
        scan_msg.width        = np.size(point_data, 0)
        scan_msg.height       = 1
        scan_msg.row_step     = scan_msg.width * scan_msg.point_step
        scan_msg.is_bigendian = True
        scan_msg.is_dense     = True

        scan_msg.fields = [
            PointField(name='x'           , offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y'           , offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z'           , offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name=last_dim_name , offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        return scan_msg
    
    def prepare_LaserScan_msg(self, ranges, intensities, angle_increment_deg, range_min, range_max, header):
        # Prepare data to be published as ros message
        scan_msg = LaserScan()

        scan_msg.header = header

        scan_msg.angle_min = 0.0
        scan_msg.angle_max = 2 * math.pi
        scan_msg.scan_time = 0.0
        scan_msg.time_increment = 0.0
        
        scan_msg.range_min = range_min
        scan_msg.range_max = range_max
        scan_msg.angle_increment = (angle_increment_deg * math.pi / 180)
        
        scan_msg.ranges = ranges.tolist()
        scan_msg.intensities = intensities.tolist()

        return scan_msg

    def cleanup(self):
        pass

def main():
    node = ToFPreProcess()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
