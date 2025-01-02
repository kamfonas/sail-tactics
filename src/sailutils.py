
#%% Load libraries
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.interpolate import CubicSpline
# from pyproj import Geod

# # Initialize a global Geod object
# geod = Geod(ellps="WGS84")


#%% Functions to Load and Filter data
def load_boat_data(file_path):
    """
    Load boat data from a JSON file.
    :param file_path: Path to the file containing boat data.
    :return: List of boat data dictionaries.
    """
    with open(file_path, 'r') as file:
        return json.load(file)
    
def filter_boats(boats, sailnumber=None, name=None,builder=None, boat_type=None):
    """
    Filter boats by sailnumber, builder, and/or type with case-insensitive partial matching.
    Returns full dictionaries of matching boats.
    :param boats: List of boat dictionaries.
    :param sailnumber: Partial sailnumber to filter by (case-insensitive).
    :param name: Partial name to filter by (case-insensitive).
    :param builder: Partial builder name to filter by (case-insensitive).
    :param boat_type: Partial boat type to filter by (case-insensitive).
    :return: List of dictionaries representing the matching boats.
    """
    def matches(field, keyword):
        """Helper function to check if a field matches the keyword (case-insensitive partial match)."""
        if not field or not keyword:  # Ignore None fields or empty keywords
            return False
        return keyword.lower() in field.lower()

    # Apply filters
    return [
        boat for boat in boats
        if (not sailnumber or matches(boat.get('sailnumber', ''), sailnumber)) and
           (not name or matches(boat.get('name', ''), name)) and
           (not builder or matches(boat.get('boat', {}).get('builder', ''), builder)) and
           (not boat_type or matches(boat.get('boat', {}).get('type', ''), boat_type))
    ]

def get_boat_index_by_sailnumber(boats, sailnumber):
    """
    Return the index of a boat with the given sailnumber (without an explicit loop).
    :param boats: List of boat dictionaries.
    :param sailnumber: Sailnumber to search for (exact match).
    :return: Index of the boat or None if not found.
    """
    return next((index for index, boat in enumerate(boats) if boat.get('sailnumber') == sailnumber), None)

#%% Functions for Polars
def get_polar_data(boat):
    """
    Extract polar data from a boat dictionary, including extended VMG data with beat and run extremes.
    :param boat: Dictionary containing boat data.
    :return: Dictionary with extended polar data.
    """
    vpp = boat.get('vpp', {})
    
    if not vpp:
        return None  # No VPP data available

    # Extract base polar data
    base_angles = vpp.get('angles', [])
    base_speeds = vpp.get('speeds', [])
    base_data = {angle: vpp.get(str(angle), []) for angle in base_angles}
    max_base_angle = max(base_angles)

    # Extract beat and run extremes
    beat_angles = vpp.get('beat_angle', [])
    beat_vmgs = vpp.get('beat_vmg', [])
    beat_speeds = [vmg / math.cos(math.radians(twa)) for vmg,twa in zip(beat_vmgs,beat_angles)]
    run_angles = vpp.get('run_angle', [])
    run_vmgs = vpp.get('run_vmg', [])
    run_speeds = [vmg / math.cos(math.radians(180 - twa)) for vmg,twa in zip(run_vmgs,run_angles)]

    # Extend VMG lines with extremes
    extended_vmg_lines = {}
    for i, wind_speed in enumerate(base_speeds):
        # Create a unified VMG line for this wind speed
        beat_point = (beat_angles[i], beat_speeds[i]) if i < len(beat_angles) else None
        run_point = (run_angles[i], run_speeds[i]) if i < len(run_angles)  else None
        main_points = [(angle, base_data[angle][i]) for angle in base_angles]

        # Combine points and sort by angle
        vmg_line = [p for p in [beat_point, *main_points, run_point] if p]
        extended_vmg_lines[wind_speed] = sorted(vmg_line, key=lambda x: x[0])

    return {
        'angles': base_angles,
        'speeds': base_speeds,
        'data': base_data,
        'extended_vmg': extended_vmg_lines
    }
    
#%% Polars from polar_data (half plot)

def plot_half_polar_data_cubicSpline(boat, polar_data):
    """
    Plot a half-polar diagram (0-180 degrees) for a boat using extended VMG lines with cubic interpolation.
    Also plot the given data points for evaluation.
    :param boat: Dictionary containing boat data.
    :param polar_data: Dictionary containing extended polar data.
    """
    # Extract extended VMG data
    extended_vmg = polar_data.get('extended_vmg', {})
    base_speeds = polar_data.get('speeds', [])

    # Generate the title: "[sailnumber] boat builder: type"
    name = boat.get('name', 'Unknown')
    sailnumber = boat.get('sailnumber', 'Unknown')
    builder = boat.get('boat', {}).get('builder', 'Unknown Builder')
    boat_type = boat.get('boat', {}).get('type', 'Unknown Type')
    title = f"{name}\n[{sailnumber}] {builder}: {boat_type}"

    # Initialize the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

    # Limit the plot to 0-180 degrees
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Plot extended VMG lines with cubic interpolation
    for wind_speed, vmg_line in extended_vmg.items():
        # Extract angles and speeds
        angles = [point[0] for point in vmg_line]
        speeds = [point[1] for point in vmg_line]

        # Interpolate the data for smoother lines
        cubic_interp = CubicSpline(angles, speeds, bc_type='natural')
        interpolated_angles = np.linspace(min(angles), max(angles), 500)
        interpolated_speeds = cubic_interp(interpolated_angles)

        # Convert interpolated angles to radians
        interpolated_angles_rad = np.radians(interpolated_angles)

        # Plot the interpolated line
        ax.plot(interpolated_angles_rad, interpolated_speeds, label=f"{wind_speed} knots")

        # Plot the given data points
        angles_rad = np.radians(angles)  # Convert angles to radians
        ax.scatter(angles_rad, speeds, s=20, marker='o', color=ax.get_lines()[-1].get_color())
        # ax.scatter(angles_rad, speeds, label=f"{wind_speed} knots (data)", s=20, marker='o')

    # Configure radial axis
    max_boat_speed = max(max([point[1] for point in vmg_line]) for vmg_line in extended_vmg.values())
    radial_ticks = np.arange(0, max_boat_speed + 2, 2)  # Circles every 2 knots
    ax.set_rmax(max_boat_speed)  # Set max radius to the maximum boat speed
    ax.set_rticks(radial_ticks)  # Set radial ticks to achieved boat speed values
    ax.set_rlabel_position(90)  # Place radial labels at the center
    ax.set_rgrids(radial_ticks, labels=[f"{v} kn" for v in radial_ticks])  # Label radial grids with boat speeds

    # Configure the plot
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise direction
    ax.set_title(title, va='bottom', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Show the plot
    plt.show()


#%% calc and plot course diagram with full polar for given TWS


def plot_polar_with_course_diagram(A, B, polar_data, TWS, TWD=90, method = 'VMC'):
    """
    Plot a polar diagram using TWA conventions and a Cartesian AB course diagram with wind direction.
    """
    # Extract coordinates and calculate the intended COG relative to TWD
    x_A, y_A = A
    x_B, y_B = B
    COG = np.degrees(np.arctan2(x_B - x_A,y_B - y_A)) % 360
    TWA_VMC = (COG - TWD) % 360
    if TWA_VMC > 180:
        TWA_VMC = 360 - TWA_VMC  # Reflect for polar symmetry
    print(f"COG: {COG:.2f}°, TWA_VMC (relative to wind): {TWA_VMC:.2f}°")

    # Interpolate the polar curve for the given TWS
    polar_line = interpolate_polar_curve(polar_data, TWS)
    if polar_line is None:
        print("Error: Unable to interpolate polar curve for the given TWS.")
        return

    # # Find the optimal VMC
    # best_twa, best_sow = find_optimal_vmc(polar_line, TWA_VMC)
    # best_vmc = best_sow * np.cos(np.radians(best_twa - TWA_VMC))
    # print(f"Best VMC at TWA {best_twa:.2f}°: SOG={best_sow:.2f} kn, VMC={best_vmc:.2f} kn")

    # Plot the polar diagram
    # plot_polar_diagram(polar_line, TWA_VMC, best_twa, best_sow, best_vmc, TWS)
    # optimal_values = plot_polar_diagram(polar_line, TWA_VMC, TWS)
    # plot_polar_diagram(
    #     polar_line, TWA_VMC, best_twa_port, best_sow_port, best_vmc_port, TWS
    # )
    # Plot the AB course diagram 
    plot_course_diagram_with_tack(A, B, TWD, polar_line, TWA_VMC)



def calculate_tack_or_jibe(TWAs, SOWs, A, B, TWD):
    """
    Calculate the point of tack or jibe and total time and distance for the course.

    :param TWAs: List of one or two true wind angles (degrees).
    :param SOWs: List of one or two speeds over water (knots).
    :param A: Tuple (Northing, Easting) representing the starting point (nautical miles).
    :param B: Tuple (Northing, Easting) representing the ending point (nautical miles).
    :param TWD: True Wind Direction (degrees).
    :return: Dictionary with course details, including tack/jibe point (if applicable).
    """
    # Calculate total distance from A to B
    delta_x = B[0] - A[0]  # Northing difference
    delta_y = B[1] - A[1]  # Easting difference
    distance_AB = np.sqrt(delta_x**2 + delta_y**2)
    COG = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360  # Course over ground

    if len(TWAs) == 1:
        # Single leg case
        time = distance_AB / SOWs[0]
        return {
            "total_distance": distance_AB,
            "total_time": time,
            "tack_point": None,
            "course": [(A[0], A[1]), (B[0], B[1])]
        }

    elif len(TWAs) == 2:
        # Two-leg case
        TWA1, TWA2 = TWAs
        SOW1, SOW2 = SOWs

        # Convert TWA to true course direction (Northing and Easting components)
        leg1_heading = (TWD + TWA1) % 360
        leg2_heading = (TWD + TWA2) % 360

        # Calculate direction vectors for both legs
        leg1_dx = SOW1 * np.cos(np.radians(leg1_heading))
        leg1_dy = SOW1 * np.sin(np.radians(leg1_heading))
        leg2_dx = SOW2 * np.cos(np.radians(leg2_heading))
        leg2_dy = SOW2 * np.sin(np.radians(leg2_heading))

        # Calculate intersection point (tack or jibe point)
        denominator = leg1_dx * -leg2_dy - -leg2_dx * leg1_dy
        if abs(denominator) < 1e-6:
            raise ValueError("The two legs are nearly parallel and cannot intersect.")

        t = ((B[0] - A[0]) * -leg2_dy - (B[1] - A[1]) * -leg2_dx) / denominator
        tack_point = (A[0] + t * leg1_dx, A[1] + t * leg1_dy)

        # Calculate distances for each leg
        distance_leg1 = np.sqrt((tack_point[0] - A[0])**2 + (tack_point[1] - A[1])**2)
        distance_leg2 = np.sqrt((B[0] - tack_point[0])**2 + (B[1] - tack_point[1])**2)

        # Calculate total time
        time_leg1 = distance_leg1 / SOW1
        time_leg2 = distance_leg2 / SOW2
        total_time = time_leg1 + time_leg2

        return {
            "total_distance": distance_leg1 + distance_leg2,
            "total_time": total_time,
            "tack_point": tack_point,
            "course": [(A[0], A[1]), tack_point, (B[0], B[1])]
        }

    else:
        raise ValueError("Invalid number of TWAs. Must be one or two.")


def plot_course_diagram_with_tack(A, B, TWD, polar_line, TWA_VMC):
    """
    Plot the AB course diagram with two optimal heading legs, tack point, and intended course.
    """
    # Calculate the bearing (COG) and distance from A to B
    # Calculate total distance from A to B
    delta_x = B[0] - A[0]  # Northing difference
    delta_y = B[1] - A[1]  # Easting difference
    distance_AB = np.sqrt(delta_x**2 + delta_y**2)
    COG = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360  # Course over ground
    
    # Calculate VMG beat/run angles and speeds
    _ix_up = np.argmax([np.cos(np.deg2rad(a))*s for a,s in zip(polar_line[0],polar_line[1])])
    _ix_down = np.argmin([np.cos(np.deg2rad(a))*s for a,s in zip(polar_line[0],polar_line[1])])
    beat_angle, beat_speed =  polar_line[0][_ix_up], polar_line[1][_ix_up]
    run_angle, run_speed   =  polar_line[0][_ix_down], polar_line[1][_ix_down]

    
    if 0 <= TWA_VMC < beat_angle:
        TWA_vmg = [beat_angle,360-beat_angle]
        SOW_vmg = [beat_speed,beat_speed]
    elif 360-beat_angle < TWA_VMC < 360:
        TWA_vmg = [360-beat_angle,beat_angle]
        SOW_vmg = [beat_speed,beat_speed]
    elif run_angle < TWA_VMC <= 180:
        TWA_vmg = [run_angle,360-run_angle]
        SOW_vmg = [run_speed,run_speed]
    elif 360-run_angle < TWA_VMC < 180:
        TWA_vmg = [360-run_angle,run_angle]
        SOW_vmg = [run_speed,run_speed]
    else: 
        TWA_vmg = []
        SOW_vmg = []

        
    angles_port, speeds_port  = polar_line 
    angles_starboard = (360 - np.array(polar_line[0])) % 360  # Mirror over 0 axis
    speeds_starboard = polar_line[1]  # Symmetrical speeds
    # Single leg direct to the target where possible
    if  beat_angle <= TWA_VMC <= run_angle:
        _idx = np.searchsorted(angles_port, TWA_VMC) - 1
        TWA_dir = [angles_port[_idx]]
        SOW_dir = [speeds_port[_idx]]
    elif 360 - run_angle < TWA_VMC < 360 - beat_angle:
        _idx = np.searchsorted(angles_starboard, TWA_VMC) - 1
        TWA_dir = [angles_starboard[_idx]]
        SOW_dir = [speeds_starboard[_idx]]
    else:
        TWA_dir = []
        SOW_dir = []
                

    best_twa_port, best_sow_port = find_optimal_vmc(polar_line, TWA_VMC)
    best_twa_starboard, best_sow_starboard = find_optimal_vmc((angles_starboard, speeds_starboard), TWA_VMC)

    if 0 <= TWA_VMC <= 90:
        TWA_vmc = [best_twa_port,best_twa_starboard]
        SOW_vmc = [best_sow_port,best_sow_starboard]
    elif 270 <= TWA_VMC <= 360:
        TWA_vmc = [best_twa_starboard,best_twa_port]
        SOW_vmc = [best_sow_starboard,best_sow_port]
    elif 90 < TWA_VMC <= 180:
        TWA_vmc = [best_twa_port,best_twa_starboard]
        SOW_vmc = [best_sow_port,best_sow_starboard]
    elif 180 < TWA_VMC < 270:
        TWA_vmc = [best_twa_starboard,best_twa_port]
        SOW_vmc = [best_sow_starboard,best_sow_port]
    else: 
        TWA_vmc = []
        SOW_vmc = []
    
        
    # Plot the course diagram
    plt.figure(figsize=(8, 8))

    # Add the intended course AB
    plt.plot([A[1], B[1]], [A[0], B[0]], linestyle="-", color="black", label="Intended Course AB")
    plt.text((A[1] + B[1]) / 2, (A[0] + B[0]) / 2, f"Distance: {distance_AB:.2f} NM\nCOG: {COG:.2f}°",
             fontsize=10, color="blue")
    # Mark the points
    plt.scatter([A[1], B[1]], [A[0], B[0]], color="red", label="Points A, B")
    plt.text(A[1], A[0], "A", fontsize=12, color="red", ha="right")
    plt.text(B[1], B[0], "B", fontsize=12, color="red", ha="left")

    # Add wind direction arrow
    wind_dx = 5 * np.sin(np.radians(TWD))
    wind_dy = 5 * np.cos(np.radians(TWD))
    plt.arrow(A[1], A[0], wind_dx, wind_dy, head_width=0.5, head_length=0.7, fc="green", ec="green", label="True Wind Direction")


    plt.axis("equal")
    plt.legend()
    # plt.title("Course Diagram with Optimal Tacks")
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.grid()
    plt.show()


def interpolate_polar_curve(polar_data, TWS):
    """
    Interpolate the polar curve for the given TWS using cubic spline.
    """
    extended_vmg = polar_data.get("extended_vmg", {})
    base_speeds = sorted(extended_vmg.keys())
    if TWS < base_speeds[0] or TWS > base_speeds[-1]:
        print(f"Warning: TWS={TWS} is outside the polar data range ({base_speeds[0]}–{base_speeds[-1]}).")
        return None

    # Find bounding TWS values
    for i in range(len(base_speeds) - 1):
        lower_tws, upper_tws = base_speeds[i], base_speeds[i + 1]
        if lower_tws <= TWS <= upper_tws:
            break

    lower_line = extended_vmg[lower_tws]
    upper_line = extended_vmg[upper_tws]
    angles = [point[0] for point in lower_line]
    lower_speeds = [point[1] for point in lower_line]
    upper_speeds = [point[1] for point in upper_line]

    # Interpolate speeds for the given TWS
    interpolated_speeds = [
        np.interp(TWS, [lower_tws, upper_tws], [lower, upper])
        for lower, upper in zip(lower_speeds, upper_speeds)
    ]

    # Smooth the polar curve using cubic spline
    cubic_spline = CubicSpline(angles, interpolated_speeds, bc_type='natural')
    smooth_angles = np.linspace(min(angles), max(angles), 500)
    smooth_speeds = cubic_spline(smooth_angles)

    return smooth_angles, smooth_speeds


def find_optimal_vmc(polar_line, TWA_VMC=0):
    """
    Find the TWA and speed on the polar curve that maximize the VMC.
    """
    angles, speeds = polar_line
    vmcs = [speed * np.cos(np.radians(angle - TWA_VMC)) for angle, speed in zip(angles, speeds)]
    best_index = np.argmax(vmcs)
    return angles[best_index], speeds[best_index]

def plot_polar_diagram(polar_line, TWA_VMC, TWS):
    """
    Plot the polar diagram with interpolated TWS curve for both port and starboard tacks,
    VMC lines, tangent lines, and legend details. Return optimal values for use in other functions.
    """
    angles_port, speeds_port = polar_line

    # Mirror port tack to get starboard tack
    angles_starboard = (360 - np.array(angles_port)) % 360  # Mirror over 0 axis
    speeds_starboard = speeds_port  # Symmetrical speeds

    # Sort starboard tack angles for plotting
    angles_starboard_sorted = np.sort(angles_starboard)
    speeds_starboard_sorted = [speeds_starboard[i] for i in np.argsort(angles_starboard)]

    # Find VMC-maximizing directions
    best_twa_port, best_sow_port = find_optimal_vmc((angles_port, speeds_port), TWA_VMC)
    best_vmc_port = best_sow_port * np.cos(np.radians(best_twa_port - TWA_VMC))
    best_twa_starboard, best_sow_starboard = find_optimal_vmc((angles_starboard_sorted, speeds_starboard_sorted), TWA_VMC)
    best_vmc_starboard = best_sow_starboard * np.cos(np.radians(best_twa_starboard - TWA_VMC))

    print(f"Port Tack: Best TWA={best_twa_port:.2f}°, SOG={best_sow_port:.2f} kn, VMC={best_vmc_port:.2f} kn")
    print(f"Starboard Tack: Best TWA={best_twa_starboard:.2f}°, SOG={best_sow_starboard:.2f} kn, VMC={best_vmc_starboard:.2f} kn")

    # Create polar plot with increased size
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(14, 12))  # Larger figure size
    ax.set_theta_zero_location("N")  # TWA=0 at the top
    ax.set_theta_direction(-1)       # Angles increase clockwise

    # Plot port tack polar curve
    ax.plot(np.radians(angles_port), speeds_port, label=None, color="blue")

    # Plot starboard tack polar curve
    ax.plot(np.radians(angles_starboard_sorted), speeds_starboard_sorted, label=None, color="green")
    # ax.plot(np.radians(angles_starboard_sorted), speeds_starboard_sorted, label="Starboard Tack Polar Curve", color="green")

    # Plot VMC lines and tangent lines
    def plot_vmc_and_tangent(ax, TWA_VMC, best_twa, best_sow, best_vmc, color, label_prefix):
        # VMC Line
        _q = np.cos(np.radians(best_twa - TWA_VMC))
        TWA_VMC = (TWA_VMC + 180) if _q < 0 else TWA_VMC
        _q = abs(_q)
        vmc_line_length = np.abs(best_sow * _q)
        ax.plot([0, np.radians(TWA_VMC)], [0, best_sow], linestyle="--", color=color,
                label=f"{label_prefix} VMC={best_vmc:.2f} kn at TWA {TWA_VMC:.2f}°")

        # Tangent Line
        ax.plot([np.radians(best_twa), np.radians(TWA_VMC)], [best_sow, vmc_line_length],
                linestyle=":", color=color, label=f"{label_prefix} Optimal Heading (TWA={best_twa:.2f}°)")

    # Port Tack
    plot_vmc_and_tangent(ax, TWA_VMC, best_twa_port, best_sow_port, best_vmc_port, "orange", "Port Tack")

    # Starboard Tack
    plot_vmc_and_tangent(ax, TWA_VMC, best_twa_starboard, best_sow_starboard, best_vmc_starboard, "red", "Starboard Tack")

    # Highlight optimal points
    ax.scatter(np.radians(best_twa_port), best_sow_port, color="darkblue", label="Optimal Point (Port Tack)")
    ax.scatter(np.radians(best_twa_starboard), best_sow_starboard, color="darkgreen", label="Optimal Point (Starboard Tack)")

    # Add title with padding
    # ax.set_title(f"Sailboat Polar Diagram (TWS={TWS} kn)\nTWA=0 at Top", va='bottom', pad=40)

    # Adjust legend placement
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=10)  # Move legend below the plot

    # Configure radial ticks and labels
    ax.set_rmax(max(speeds_port) + 2)
    ax.set_rticks(np.arange(0, max(speeds_port) + 2, 2))
    ax.set_rlabel_position(90)

    # Tighten layout and leave space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at the bottom for the legend

    plt.show()

    # Return the optimal values
    return best_twa_port, best_sow_port, best_vmc_port, best_twa_starboard, best_sow_starboard, best_vmc_starboard


