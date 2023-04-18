import numpy as np
import time
import sys
import os
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2

'''
If not working, make sure you've downloaded whl file
https://artifactory.kinovaapps.com/ui/api/v1/download?repoKey=generic-public&path=kortex%2FAPI%2F2.3.0%2Fkortex_api-2.3.0.post34-py3-none-any.whl 
and run python -m pip install <whl relative fullpath name>.whl in terminal

Make sure utilities is set to correct robot (all sequences were run on 192.168.1.10 
'''

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 30

# Create closure to set an event after an END or an ABORT
def check_for_sequence_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications on a sequence

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e = e):
        event_id = notification.event_identifier
        task_id = notification.task_index
        if event_id == Base_pb2.SEQUENCE_TASK_COMPLETED:
            print("Sequence task {} completed".format(task_id))
        elif event_id == Base_pb2.SEQUENCE_ABORTED:
            print("Sequence aborted with error {}:{}"\
                .format(\
                    notification.abort_details,\
                    Base_pb2.SubErrorCodes.Name(notification.abort_details)))
            e.set()
        elif event_id == Base_pb2.SEQUENCE_COMPLETED:
            print("Sequence completed.")
            e.set()
    return check

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def init_robot(base, base_cyclic, base_feedback):
    print("Initializing")
    base_fb = SendCallWithRetry(base_cyclic.RefreshFeedback, 3)
    if base_fb:
        base_feedback = base_fb
    return base_feedback

def move_to_home(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        sys.exit(0)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)

    # Leave time to action to complete
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout on action notification wait")
    return finished

def readAngles(filepath):
    data = np.loadtxt(filepath, delimiter=",", dtype=str)
    angles = data[1:, np.arange(5, 16, 2)].astype(float)
    print(angles)
    return angles

def create_angular_action(actuator_count, action_number, angles):
    print("Creating angular action")
    action = Base_pb2.Action()
    action.name = "Action " + str(action_number)
    action.application_data = ""

    for joint_id in range(actuator_count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.value = angles[joint_id]

    return action

def create_sequence(actuator_count, seq_name, in_file):
    angles = readAngles(in_file)
    action_count = angles.shape[0]

    print("Creating Sequence")
    sequence = Base_pb2.Sequence()
    sequence.name = seq_name

    for action_number in np.arange(action_count):
        print("Creating Action " + str(action_number))
        angular_action = create_angular_action(actuator_count, action_number, angles[action_number, :])

        print("Adding Action " + str(action_number))
        task = sequence.tasks.add()
        task.group_identifier = action_number  # sequence elements with same group_id are played at the same time
        task.action.CopyFrom(angular_action)

    print("Done Creating Sequence")

    return sequence

def run_sequence(base, base_feedback, actuator_count, seq_name, in_file, out_file):
    sequence = create_sequence(actuator_count, seq_name, in_file)

    e = threading.Event()
    notification_handle = base.OnNotificationSequenceInfoTopic(
        check_for_sequence_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Creating sequence on device and executing it")
    handle_sequence = base.CreateSequence(sequence)
    base.PlaySequence(handle_sequence)

    print("Waiting for movement to finish ...")

    t_start = time.time()
    t_now = t_start
    t_cyclic = t_start
    t_sample = 0.1

    count = 0

    arm_data = []
    while (t_now - t_start) <= TIMEOUT_DURATION:
        t_now = time.time()
        if (t_now - t_cyclic) >= t_sample:
            t_cyclic = t_now
            sample_data = np.array([count])
            for actuator_num in np.arange(actuator_count):
                pos = base_feedback.actuators[actuator_num].position
                vel = base_feedback.actuators[actuator_num].velocity
                tor = base_feedback.actuators[actuator_num].torque
                sample_data = np.hstack((sample_data, np.array([pos, vel, tor])))
            count += 1
            arm_data.append(sample_data)

    arm_data = np.array(arm_data)

    heading = ("Count, J1_Pos, J1_Vel, J1_Tor, J2_Pos, J2_Vel, J2_Tor, J3_Pos, J3_Vel, J3_Tor, J4_Pos, J4_Vel, J4_Tor, "
               "J5_Pos, J5_Vel, J5_Tor, J6_Pos, J6_Vel, J6_Tor")

    np.savetxt(out_file, arm_data, delimiter=", ", header=heading, comments='')

    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if not finished:
        print("Timeout on action notification wait")
    return finished

def SendCallWithRetry(call, retry, *args):
    i = 0
    arg_out = []
    while i < retry:
        try:
            arg_out = call(*args)
            break
        except:
            i = i + 1
            continue
    if i == retry:
        print("Failed to communicate")
    return arg_out

def main():
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            # Create required services
            device_manager = DeviceManagerClient(router)
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router_real_time)

            base_feedback = BaseCyclic_pb2.Feedback()

            device_handles = device_manager.ReadAllDevices()
            actuator_count = base.GetActuatorCount().count

            for handle in device_handles.device_handle:
                if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                    base_feedback.actuators.add()

            base_feedback = init_robot(base, base_cyclic, base_feedback)

            success = True
            # names = ['Seq1', 'Seq2', 'Seq3', 'Seq4', 'Seq5']
            # timeout durations = [30, 40, 70, 100, 60]
            name = 'Seq1'
            in_file = './Input_Seq/Input_' + name + '.csv'
            out_file = './Output_Seq/Output_Test_' + name + '.csv'

            success &= move_to_home(base)
            success &= run_sequence(base, base_feedback, actuator_count, name, in_file, out_file)

            return 0 if success else 1

if __name__ == "__main__":
    exit(main())