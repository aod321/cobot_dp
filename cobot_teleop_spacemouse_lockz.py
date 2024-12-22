import multiprocessing as mp
from multiprocessing import Queue
import time
from dataclasses import dataclass
import numpy as np
from rm_65_model import RM65
from cobot_controller import RobotController
from spatialmath import SE3
import roboticstoolbox as rtb
import pyspacemouse
import cv2
import scipy.spatial.transform as st
import scipy.interpolate as si
from typing import Union
import numbers
import enum

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:,:3]
            rot = st.Rotation.from_rotvec(poses[:,3:])

            self.pos_interp = si.interp1d(times, pos, 
                axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:,:3] = self.pos_interp(t)
            pose[:,3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose

def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    GRIPPER = 3
    CLEAR = 4

@dataclass
class MoveCommand:
    direction: str  # 'pose', 'gripper', 'estop', 'clear'
    value: float
    timestamp: float  # 时间戳
    target_time: float = None  # 目标执行时间

class SpaceMouseController(mp.Process):
    def __init__(self, command_queue: Queue, direct_command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue # 用于位置/姿态控制
        self.direct_command_queue = direct_command_queue # 用于clear和gripper控制
        self.linear_scale = 0.001  # 线性移动缩放因子 (meters)
        self.angular_scale = 0.001  # 角度移动缩放因子 (radians)
        self.gripper_state = False
        self.button_debounce_time = 0.2  # 按钮防抖时间(秒)
        self.last_button_press = 0
        self.lock_z = False  # 锁定z轴和旋转的标志
        self.deadzone = 0.0001  # 增加死区阈值变量
        self.control_period = 0.01  # 控制周期10ms
        
    def run(self):
        success = pyspacemouse.open()
        if not success:
            print("Failed to connect to SpaceMouse!")
            return
            
        print("SpaceMouse control started")
        print("Button 0: Toggle gripper")
        print("Button 1: Emergency stop")
        print("Press 'l' to toggle lock z-axis and rotation")
        
        cv2.namedWindow('Keyboard Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Keyboard Control', 1, 1)
        
        next_cycle_time = time.monotonic()
        
        while True:
            current_time = time.monotonic()
            
            # 等待直到下一个控制周期
            precise_wait(next_cycle_time)
            next_cycle_time = current_time + self.control_period
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'):
                self.lock_z = not self.lock_z
                print(f"Z-axis and rotation {'locked' if self.lock_z else 'unlocked'}")
            
            state = pyspacemouse.read()
            if state:
                # 应用死区
                x = state.x if abs(state.x) > self.deadzone else 0
                y = state.y if abs(state.y) > self.deadzone else 0
                z = state.z if abs(state.z) > self.deadzone else 0
                roll = state.roll if abs(state.roll) > self.deadzone else 0
                pitch = state.pitch if abs(state.pitch) > self.deadzone else 0
                yaw = state.yaw if abs(state.yaw) > self.deadzone else 0
                
                if self.lock_z:
                    pose_delta = [
                        -y * self.linear_scale,
                        x * self.linear_scale,
                        0,
                        0,
                        0,
                        0
                    ]
                else:
                    pose_delta = [
                        -y * self.linear_scale,
                        x * self.linear_scale,
                        z * self.linear_scale,
                        roll * self.angular_scale,
                        pitch * self.angular_scale,
                        yaw * self.angular_scale
                    ]
                
                threshold = 0.00001
                if any(abs(v) > threshold for v in pose_delta):
                    target_time = current_time + self.control_period
                    command = {
                        'cmd': Command.SERVOL.value,
                        'target_pose': pose_delta,
                        'duration': self.control_period,
                        'target_time': target_time
                    }
                    self.command_queue.put(command)
                
                if current_time - self.last_button_press > self.button_debounce_time:
                    if state.buttons[0]:
                        self.gripper_state = not self.gripper_state
                        command = {
                            'cmd': Command.GRIPPER.value,
                            'value': float(self.gripper_state),
                            'target_time': current_time + self.control_period
                        }
                        self.direct_command_queue.put(command)
                        self.last_button_press = current_time
                    elif state.buttons[1]:
                        command = {
                            'cmd': Command.CLEAR.value,
                            'target_time': current_time + self.control_period
                        }
                        self.direct_command_queue.put(command)
                        self.last_button_press = current_time

class RobotControlProcess(mp.Process):
    def __init__(self, command_queue: Queue, direct_command_queue: Queue):
        super().__init__()
        self.command_queue = command_queue # 用于位置/姿态控制
        self.direct_command_queue = direct_command_queue # 用于clear和gripper控制
        self.current_pose = None
        self.current_joints = None
        self.current_joints_rad = None
        self.control_period = 0.01  # 控制周期10ms
        self.command_buffer = []  # 存储带时间戳的命令
        self.max_pos_speed = 0.25  # m/s
        self.max_rot_speed = 0.16  # rad/s
        self.pose_interp = None

    def run(self):
        self.cobot_controller = RobotController()
        self.rm_65_ik_model = RM65()
        self.cobot_controller.connect()
        self.cobot_controller.speed_upbound = 50
        self.update_current_pose()
        
        print("机械臂控制已启动")
        
        next_cycle_time = time.monotonic()
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        self.pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[self.current_pose]
        )
        
        while True:
            current_time = time.monotonic()
            
            # 等待直到下一个控制周期
            precise_wait(next_cycle_time)
            next_cycle_time = current_time + self.control_period
            
            # 处理直接命令(clear和gripper)
            try:
                while True:
                    command = self.direct_command_queue.get_nowait()
                    if command['cmd'] == Command.CLEAR.value:
                        self.cobot_controller.clear_error()
                        self.cobot_controller.resume()
                        self.update_current_pose()
                    elif command['cmd'] == Command.GRIPPER.value:
                        if command['value'] == 1:
                            self.cobot_controller.gripper.open()
                        elif command['value'] == 0:
                            self.cobot_controller.gripper.close()
            except:
                pass
            
            # 获取所有可用的新位置/姿态命令并按时间戳排序
            try:
                while True:
                    command = self.command_queue.get_nowait()
                    if command['cmd'] == Command.SERVOL.value:
                        target_pose = self.current_pose + np.array(command['target_pose'])
                        duration = float(command['duration'])
                        curr_time = current_time + self.control_period
                        t_insert = curr_time + duration
                        self.pose_interp = self.pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
            except:
                pass

            # 获取当前时刻的插值位姿并发送给机器人
            t_now = time.monotonic()
            pose_command = self.pose_interp(t_now)
            self.cobot_controller.movep_canfd(pose_command.tolist(), follow=False)
            self.current_pose = pose_command

    def update_current_pose(self):
        deg2rad = np.pi/180
        self.current_pose = self.cobot_controller.calc_end_pose()
        self.current_joints = self.cobot_controller.get_joint_angles()
        self.current_joints_rad = np.array(self.current_joints) * deg2rad

def main():
    command_queue = Queue()
    direct_command_queue = Queue()
    
    spacemouse_controller = SpaceMouseController(command_queue, direct_command_queue)
    robot_controller = RobotControlProcess(command_queue, direct_command_queue)
    
    spacemouse_controller.start()
    robot_controller.start()
    
    spacemouse_controller.join()
    robot_controller.terminate()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()