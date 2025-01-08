#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Pose
from rm_msgs.msg import CartePos
from std_msgs.msg import Empty, Bool
from dh_gripper_msgs.msg import GripperCtrl
import pyspacemouse
import time
import sys
import tf.transformations
import numpy as np

class SpaceMouseControl:
    def __init__(self):
        rospy.init_node('spacemouse_control', anonymous=True)
        
        # 创建发布者，发布末端位姿命令
        self.pub = rospy.Publisher('/rm_driver/MoveP_Fd_Cmd', CartePos, queue_size=10)
        
        # 订阅当前位姿
        self.pose_sub = rospy.Subscriber('/rm_driver/Pose_State', Pose, self.pose_callback)
        
        # 创建清除系统错误的发布者
        self.clear_system_err_pub = rospy.Publisher('/rm_driver/Clear_System_Err', Empty, queue_size=10)
        
        # 创建Gripper控制发布者
        self.gripper_pub = rospy.Publisher('/gripper/ctrl', GripperCtrl, queue_size=10, latch=True)
        
        # 实际位姿(UDP回传)
        self.current_pose = None
        # 目标位姿(spacemouse控制)
        self.target_pose = None
        
        # 移动速度缩放因子
        self.linear_scale = 0.0005  # 线性移动缩放因子 (米)
        self.angular_scale = 0.0001  # 角度移动缩放因子 (弧度)
        
        # 按钮防抖时间
        self.button_debounce_time = 0.2
        self.last_button_press = 0
        
        # 夹爪状态
        self.gripper_state = False
        
        # 等待获取初始位姿
        rospy.loginfo("等待获取机械臂当前位姿...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
            
        # 初始化目标位姿为当前位姿
        self.reset_target_pose()
        
        # 初始化spacemouse
        success = pyspacemouse.open()
        if not success:
            rospy.logerr("无法打开SpaceMouse设备!")
            sys.exit(1)
        rospy.loginfo("SpaceMouse设备已连接")

    def pose_callback(self, msg):
        """位姿话题回调函数"""
        self.current_pose = msg

    def reset_target_pose(self):
        """重置目标位置和姿态到当前位置"""
        if self.current_pose is not None:
            self.target_pose = Pose()
            self.target_pose.position = self.current_pose.position
            self.target_pose.orientation = self.current_pose.orientation
            rospy.loginfo("已重置目标位置和姿态到当前位置")

    def clear_system_error_and_reset(self):
        """清除系统错误并重置位置"""
        # 清除系统错误
        self.clear_system_err_pub.publish(Empty())
        rospy.loginfo("已发送清除系统错误指令")
        
        # 重置位置
        self.reset_target_pose()

    def control_gripper(self, action=None):
        """控制夹爪"""
        gripper_msg = GripperCtrl()
        gripper_msg.initialize = False
        
        if action is None:
            # 切换夹爪状态
            self.gripper_state = not self.gripper_state
        elif action == 'open':
            self.gripper_state = True
        elif action == 'close':
            self.gripper_state = False
        
        if self.gripper_state:
            gripper_msg.position = 1000.0
            gripper_msg.force = 0.0
            rospy.loginfo("打开夹爪")
        else:
            gripper_msg.position = 0.0
            gripper_msg.force = 100.0
            rospy.loginfo("关闭夹爪")
        
        gripper_msg.speed = 0.0
        self.gripper_pub.publish(gripper_msg)

    def publish_pose(self, delta_pose):
        """发布位姿消息"""
        # 更新目标位置
        self.target_pose.position.x += delta_pose[0]
        self.target_pose.position.y += delta_pose[1]
        self.target_pose.position.z += delta_pose[2]
        
        # 更新目标姿态
        # 获取当前姿态的四元数
        current_quat = [
            self.current_pose.orientation.w,
            self.current_pose.orientation.x, 
            self.current_pose.orientation.y, 
            self.current_pose.orientation.z
        ]
        
        # 创建增量旋转四元数
        rotate_x = tf.transformations.quaternion_about_axis(delta_pose[3], (1,0,0))
        rotate_y = tf.transformations.quaternion_about_axis(delta_pose[4], (0,1,0))
        rotate_z = tf.transformations.quaternion_about_axis(delta_pose[5], (0,0,1))
        
        # 组合旋转
        combined_rotate = tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(rotate_z, rotate_y),
            rotate_x
        )
        
        # 应用旋转
        new_quat = tf.transformations.quaternion_multiply(current_quat, combined_rotate)
        
        # 更新目标姿态
        self.target_pose.orientation.w = new_quat[0]
        self.target_pose.orientation.x = new_quat[1]
        self.target_pose.orientation.y = new_quat[2]
        self.target_pose.orientation.z = new_quat[3]
        
        # 发布消息
        msg = CartePos()
        msg.Pose = self.target_pose
        self.pub.publish(msg)

    def run(self):
        """主循环"""
        rospy.loginfo("""
控制说明:
---------------------------
移动控制:
    推动手柄控制XYZ轴移动和姿态旋转
    
按键功能:
    按钮0: 开关夹爪
    按钮1: 清除系统错误并重置位置
    
当前位置 - X: %.3f, Y: %.3f, Z: %.3f
---------------------------
开始控制!
        """, 
        self.current_pose.position.x,
        self.current_pose.position.y,
        self.current_pose.position.z)
        
        rate = rospy.Rate(100)  # 100Hz控制频率
        while not rospy.is_shutdown():
            try:
                # 读取spacemouse状态
                state = pyspacemouse.read()
                if state:
                    # 计算当前时间
                    current_time = time.time()
                    
                    # 构建位姿增量
                    pose_delta = [
                        state.x * self.linear_scale,     # x
                        state.y * self.linear_scale,     # y
                        state.z * self.linear_scale,     # z
                        state.roll * self.angular_scale,   # rx
                        state.pitch * self.angular_scale,  # ry
                        state.yaw * self.angular_scale     # rz
                    ]
                    
                    # 检查是否有显著运动
                    threshold = 0.00001
                    if any(abs(v) > threshold for v in pose_delta):
                        self.publish_pose(pose_delta)
                        rospy.loginfo(
                            "位置变化 - X: %.3f, Y: %.3f, Z: %.3f | 姿态变化 - RX: %.3f, RY: %.3f, RZ: %.3f", 
                            pose_delta[0], pose_delta[1], pose_delta[2],
                            pose_delta[3], pose_delta[4], pose_delta[5]
                        )
                    
                    # 处理按钮输入（带防抖）
                    if current_time - self.last_button_press > self.button_debounce_time:
                        if state.buttons[0]:  # 按钮0切换夹爪
                            self.control_gripper()
                            self.last_button_press = current_time
                        elif state.buttons[1]:  # 按钮1清除错误并重置
                            self.clear_system_error_and_reset()
                            self.last_button_press = current_time
                
                rate.sleep()
                
            except KeyboardInterrupt:
                break

if __name__ == '__main__':
    try:
        controller = SpaceMouseControl()
        controller.run()
    except rospy.ROSInterruptException:
        pass
