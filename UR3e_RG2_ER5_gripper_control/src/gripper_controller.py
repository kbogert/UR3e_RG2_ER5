#!/usr/bin/env python

"""Controller node for the rg2 gripper."""

import sys

import rospy
import rosgraph
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandFeedback, GripperCommandResult
from control_msgs.msg import JointControllerState
from std_msgs.msg import Float64



class GripperActionServer:
    """The action server that handles gripper commands."""

    def __init__(self,
                 servo_namespace,               
                 closed_dist,
                 open_dist,
                 timeout,
                 is_simulated):
        """Initialise the gripper action server."""

        

        self._closed_dist = closed_dist
        self._open_dist = open_dist
        self._timeout = timeout
        rospy.loginfo(
            'gripper action server configuration:\n' +          
            '\tClosed dist: {} rad\n'.format(self._closed_dist) +
            '\tOpen dist: {} rad\n'.format(self._open_dist) +
            '\tMovement timeout: {} s\n'.format(self._timeout) +
            '\tIs simulated: {}'.format(is_simulated))

        self._as = actionlib.SimpleActionServer(
            rosgraph.names.ns_join(rospy.get_name(), 'gripper_command'),
            GripperCommandAction,
            auto_start=False)
        self._as.register_goal_callback(self._handle_command)

        self._command_pub = rospy.Publisher(
            rosgraph.names.ns_join(servo_namespace, 'command'),
            Float64,
            queue_size=1)

        self._as.start()
        rospy.loginfo('Gripper action server waiting for goals')



    def _handle_command(self):
        self._goal = self._as.accept_new_goal()     
        # Start the gripper servo moving to the goal position
        self._command_pub.publish(self._goal_dist)
        # Set up the timer to prevent running forever
        self._timer = rospy.Timer(rospy.Duration(self._timeout),
                                  self._timed_out,
                                  oneshot=True)


def main():
    """Main function."""
    rospy.init_node('gripper_controller')
    servo_namespace = rospy.get_param(
        '~servo_namespace',
        '/gripper_controller')

    # The dist where the gripper is fully closed is 0 mm by default
    closed_dist = rospy.get_param('~servo_closed_dist', 0.0)
    # The dist where the gripper is fully open is 110 mm by default
    open_dist = rospy.get_param('~servo_open_dist', 110.0)
    # Default timeout of 5 seconds
    timeout = rospy.get_param('~timeout', 5)
    # Set to true if using the simulated arm
    is_simulated = rospy.get_param('~is_simulated', False)

    server = GripperActionServer(servo_namespace,
                                 closed_dist,
                                 open_dist,
                                 timeout,
                                 is_simulated)
    rospy.spin()
    return 0


if __name__ == '__main__':
    sys.exit(main())
