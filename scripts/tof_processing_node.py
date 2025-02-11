#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, LaserScan
from std_msgs.msg import Header

DEFAULT_INPUT_TOPIC = "/sensors/tof_sensors/pcl_raw"
DEFAULT_OUTPUT_TOPIC_PRE_SLICE = "/sensots/tof_sensors/pcl_pre_process"
DEFAULT_OUTPUT_TOPIC_2D_SLICE = "/sensots/tof_sensors/pcl_2d_slice"
DEFAULT_OUTPUT_TOPIC_POST_SLICE = "/sensots/tof_sensors/pcl_post_process"
DEFAULT_OUTPUT_TOPIC_COMBINED = "/sensots/tof_hybrid/hybrid_scan"
DEFAULT_PUBLISH_INTERMEDIATE_TOPICS = False
DEFAULT_FILTER_FLOOR = True
DEFAULT_FLOOR_BELOW_BASE_THRESHOLD_M = 0.03
DEFAULT_FILTER_HIGH_SIGMA = False
DEFAULT_HIGH_SIGMA_THRESHOLD = 6e-03

class ToFPreProcess:
    def __init__(self):
        rospy.init_node('tof_pre_process_node', anonymous=True)
        rospy.loginfo("Starting tof pre-processing node")

        self.INPUT_TOPIC = rospy.get_param('~INPUT_TOPIC', DEFAULT_INPUT_TOPIC)
        self.OUTPUT_TOPIC_PRE_SLICE = rospy.get_param('~INPUT_TOPIC', DEFAULT_OUTPUT_TOPIC_PRE_SLICE)
        self.OUTPUT_TOPIC_2D_SLICE = rospy.get_param('~INPUT_TOPIC', DEFAULT_OUTPUT_TOPIC_2D_SLICE)
        self.OUTPUT_TOPIC_POST_SLICE = rospy.get_param('~INPUT_TOPIC', DEFAULT_OUTPUT_TOPIC_POST_SLICE)
        self.OUTPUT_TOPIC_COMBINED = rospy.get_param('~INPUT_TOPIC', DEFAULT_OUTPUT_TOPIC_COMBINED)
        self.PUBLISH_INTERMEDIATE_TOPICS = rospy.get_param('~INPUT_TOPIC', DEFAULT_PUBLISH_INTERMEDIATE_TOPICS)
        self.FILTER_FLOOR = rospy.get_param('~FILTER_FLOOR', DEFAULT_FILTER_FLOOR)
        self.FLOOR_BELOW_BASE_THRESHOLD_M = rospy.get_param('~FLOOR_BELOW_BASE_THRESHOLD_M', DEFAULT_FLOOR_BELOW_BASE_THRESHOLD_M)
        self.FILTER_HIGH_SIGMA = rospy.get_param('~FILTER_HIGH_SIGMA', DEFAULT_FILTER_HIGH_SIGMA)
        self.HIGH_SIGMA_THRESHOLD = rospy.get_param('~HIGH_SIGMA_THRESHOLD', DEFAULT_HIGH_SIGMA_THRESHOLD)

        rospy.loginfo("INPUT_TOPIC: {}".format(self.INPUT_TOPIC))
        rospy.loginfo("OUTPUT_TOPIC_PRE_SLICE: {}".format(self.OUTPUT_TOPIC_PRE_SLICE))
        rospy.loginfo("OUTPUT_TOPIC_2D_SLICE: {}".format(self.OUTPUT_TOPIC_2D_SLICE))
        rospy.loginfo("OUTPUT_TOPIC_POST_SLICE: {}".format(self.OUTPUT_TOPIC_POST_SLICE))
        rospy.loginfo("OUTPUT_TOPIC_COMBINED: {}".format(self.OUTPUT_TOPIC_COMBINED))
        rospy.loginfo("PUBLISH_INTERMEDIATE_TOPICS: {}".format(self.PUBLISH_INTERMEDIATE_TOPICS))
        rospy.loginfo("FILTER_FLOOR: {}".format(self.FILTER_FLOOR))
        rospy.loginfo("FLOOR_BELOW_BASE_THRESHOLD_M: {}".format(self.FLOOR_BELOW_BASE_THRESHOLD_M))
        rospy.loginfo("FILTER_HIGH_SIGMA: {}".format(self.FILTER_HIGH_SIGMA))
        rospy.loginfo("HIGH_SIGMA_THRESHOLD: {}".format(self.HIGH_SIGMA_THRESHOLD))

        self.tof_sub = rospy.Subscriber(self.INPUT_TOPIC, PointCloud2, self.receive_pointcloud)
        self.tof_pre_process_pub = rospy.Publisher(self.OUTPUT_TOPIC_PRE_SLICE, PointCloud2, queue_size=10)
        self.tof_2d_slice_pub = rospy.Publisher(self.OUTPUT_TOPIC_2D_SLICE, PointCloud2, queue_size=10)
        self.tof_post_process_pub = rospy.Publisher(self.OUTPUT_TOPIC_POST_SLICE, PointCloud2, queue_size=10)
        self.tof_combined_pub = rospy.Publisher(self.OUTPUT_TOPIC_COMBINED, LaserScan, queue_size=10)

    def receive_pointcloud(self, msg):
        msg_bytes = np.frombuffer(msg.data, dtype=np.float32)
        point_cloud = np.reshape(msg_bytes, (msg.width, 4))

        filtered_idx = np.any(point_cloud)

        if self.FILTER_FLOOR:
            filtered_idx_floor = point_cloud[:, 2] > self.FLOOR_BELOW_BASE_THRESHOLD_M
            filtered_idx = np.multiply(filtered_idx, filtered_idx_floor)

        if self.FILTER_HIGH_SIGMA:
            filtered_idx_sigma = point_cloud[:, 3] < self.HIGH_SIGMA_THRESHOLD
            filtered_idx = np.multiply(filtered_idx, filtered_idx_sigma)

        filtered_points = point_cloud[filtered_idx, :]
        scan_msg_filter = self.prepare_PointCloud2_msg(filtered_points, msg.header)
        self.tof_pre_process_pub.publish(scan_msg_filter)

    def prepare_PointCloud2_msg(self, point_data, header):
        scan_msg = PointCloud2()
        scan_msg.header = header
        scan_msg.data = point_data.flatten().tobytes()
        scan_msg.point_step = 4 * 4
        scan_msg.width = np.size(point_data, 0)
        scan_msg.height = 1
        scan_msg.row_step = scan_msg.width * scan_msg.point_step
        scan_msg.is_bigendian = False
        scan_msg.is_dense = True

        scan_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='sigma', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
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
