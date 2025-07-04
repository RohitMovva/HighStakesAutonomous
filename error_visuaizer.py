import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt

def parse_log_file(file_path):
    """
    Parse the robot log file to extract pose and goal information.
    """
    poses = []  # List to store (timestamp, x, y, theta) tuples for actual poses
    goals = []  # List to store (timestamp, x, y, theta) tuples for goal poses
    
    # Regular expressions to match pose and goal lines
    pose_pattern = r'\[(\d+:\d+:\d+\.\d+)\] Pose: (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)'
    goal_pattern = r'\[(\d+:\d+:\d+\.\d+)\] Goal: (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)'
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Check for pose entries
                seconds = 0.0  # Initialize seconds to avoid reference before assignment
                pose_match = re.match(pose_pattern, line)
                if pose_match:
                    timestamp_str, x, y, theta = pose_match.groups()
                    timestamp = dt.datetime.strptime(timestamp_str, '%H:%M:%S.%f')
                    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1000000
                    poses.append((seconds, float(x), float(y), float(theta)))
                    continue
                
                # Check for goal entries
                goal_match = re.match(goal_pattern, line)
                if goal_match:
                    timestamp_str, x, y, theta = goal_match.groups()
                    timestamp = dt.datetime.strptime(timestamp_str, '%H:%M:%S.%f')
                    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1000000

                    goals.append((seconds, float(x), float(y), float(theta)))

                if (seconds > 9.8):
                    break
        
        return poses, goals
    
    except Exception as e:
        print(f"Error reading log file: {e}")
        return [], []

def calculate_errors(poses, goals):
    """
    Calculate positional and heading errors based on pose and goal data.
    
    Returns:
    - timestamps: List of timestamps
    - position_errors: List of Euclidean distance errors
    - heading_errors: List of heading angle errors (in radians)
    """
    timestamps = []
    position_errors = []
    heading_errors = []
    
    # Create a dictionary of goals indexed by timestamp for easy lookup
    goal_dict = {g[0]: (g[1], g[2], g[3]) for g in goals}
    
    for pose in poses:
        timestamp, px, py, ptheta = pose
        
        # Find the closest goal entry
        if timestamp in goal_dict:
            gx, gy, gtheta = goal_dict[timestamp]
            
            # Calculate position error (Euclidean distance)
            pos_error = math.sqrt((px - gx)**2 + (py - gy)**2)
            
            # Calculate heading error, normalizing to [-pi, pi]
            heading_error = ptheta - gtheta
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            timestamps.append(timestamp)
            position_errors.append(pos_error)
            heading_errors.append(heading_error)
    
    return timestamps, position_errors, heading_errors

def plot_errors(timestamps, position_errors, heading_errors):
    """
    Create plots for positional and heading errors over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert timestamps to seconds from first timestamp for x-axis
    if timestamps:
        # start_time = timestamps[0]
        print(f"First timestamp: {timestamps[0]}")
        time_seconds = [(t)for t in timestamps]
    else:
        time_seconds = []
    
    # Plot position error
    ax1.plot(time_seconds, position_errors, 'b-', linewidth=2)
    ax1.set_title('Robot Position Error Over Time')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Position Error (inches)')
    ax1.grid(True)
    
    # Plot heading error
    ax2.plot(time_seconds, [err * 180/math.pi for err in heading_errors], 'r-', linewidth=2)
    ax2.set_title('Robot Heading Error Over Time')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Heading Error (degrees)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('robot_error_plots.png')
    plt.show()

def main():
    log_file_path = 'robot_log_19700101_000000.txt'
    
    print(f"Parsing log file: {log_file_path}")
    poses, goals = parse_log_file(log_file_path)
    
    print(f"Found {len(poses)} pose entries and {len(goals)} goal entries")
    
    if not poses or not goals:
        print("No data found in the log file")
        return
    
    print("Calculating errors...")
    timestamps, position_errors, heading_errors = calculate_errors(poses, goals)
    
    print(f"Plotting {len(timestamps)} data points...")
    plot_errors(timestamps, position_errors, heading_errors)
    
    print(f"Plots generated. Average position error: {np.mean([abs(p) for p in position_errors]):.4f}")
    print(f"Average heading error: {np.mean([abs(h) for h in heading_errors]) * 180/math.pi:.4f} degrees")

if __name__ == "__main__":
    main()