#!/usr/bin/env python

"""Control the width that a gripper is open."""

import argparse
import sys

import rospy
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        'Send gripper width commands to a GripperActionServer')
    parser.add_argument(
        'width',
        type=float,
        help='the width (in mm) to set the gripper to')
    parser.add_argument(
        '-n',
        '--namespace',
        type=str,
        default='/gripper_command',
        help='the namespace containing the GripperCommand action')
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('gripper_command_client')
    client = actionlib.SimpleActionClient(args.namespace, GripperCommandAction)
    client.wait_for_server()
    goal = GripperCommandGoal()
    goal.command.position = args.width
    client.send_goal(goal, feedback_cb=_feedback_cb)

    client.wait_for_result()



if __name__ == '__main__':
    sys.exit(main())
