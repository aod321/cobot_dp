import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from pose_trajectory_interpolator import PoseTrajectoryInterpolator
from cobot_controller import RobotController
from precise_sleep import precise_sleep,precise_wait
from rm_65_model import RM65

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class InterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
        """

    def __init__(self,
                shm_manager: SharedMemoryManager,
                robot_ip="192.168.40.102",
                robot_port=8080,
                frequency=100,
                max_pos_speed=0.25,
                max_rot_speed=0.16,
                launch_timeout=3,
                verbose=False,
                receive_keys=None,
                get_max_k=128,
                lookahead_time=0.1,
                gain=300,
                robot_ik_model=RM65(),
                ):
        """
        frequency: CB2=125, UR3e=500, 
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.robot_ik_model = robot_ik_model

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        example = dict()
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    def get_current_pose(self):
        deg2rad = np.pi/180
        current_joints = self.get_joint_angles()
        current_joints_rad = np.array(current_joints) * deg2rad
        current_pose = self.robot_ik_model.fkine(current_joints_rad).t.tolist() + self.robot_ik_model.fkine(current_joints_rad).rpy().tolist()
        return current_pose
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # Initialize robot connection
        robot = RobotController(ip=self.robot_ip, port=self.robot_port)
        robot.speed_upbound = 50
        if not robot.connect():
            print("Failed to connect to robot")
            return
        try:
            # main loop
            dt = 1. / self.frequency
            deg2rad = np.pi/180
            curr_joints_angles = robot.get_joint_angles() # in degree
            curr_joints_rad = np.array(curr_joints_angles) * deg2rad
            current_end_pose = robot.get_end_pose()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[current_end_pose]
            )
            
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = robot.init_period()

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)
                vel = 30
                assert robot.moveL(pose_command,speed=vel)
                
                # update robot state
                # state = dict()
                # for key in self.receive_keys:
                    # state[key] = np.array(getattr(robot, 'get'+key)())
                # state['robot_receive_timestamp'] = time.time()
                # self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                precise_wait(t_start+dt)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            robot.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}")