
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

        # remove items with whith duplicate angles
        angles_dedup, unique_indices = np.unique(angles, return_index=True)
        speeds_dedup = np.array(speeds)[unique_indices]

        # Interpolate the data for smoother lines
        cubic_interp = CubicSpline(angles_dedup, speeds_dedup, bc_type='natural')
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

    if len(TWAs) == 0:
        return None
 
    if len(TWAs) == 1:
        # Single leg case
        time = distance_AB / SOWs[0]
        return {
            "distances": [distance_AB],
            "speeds":   [SOWs[0]],
            "elapsedHrs": [time],
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
            "distances": [distance_leg1, distance_leg2],
            "speeds": [SOW1,SOW2],
            "elapsedHrs": [time_leg1,time_leg2],
            "tack_point": tack_point,
            "course": [(A[0], A[1]), tack_point, (B[0], B[1])]
        }

    else:
        raise ValueError("Invalid number of TWAs. Must be none, one or two.")


def plot_course_diagram_with_tack(A, B, TWD, polar_line, TWA_VMC, vmc_data):
    """
    Plot the AB course diagram with two optimal heading legs, tack point, and intended course.
    """
    # Calculate the bearing (COG) and distance from A to B
    # Calculate total distance from A to B
    delta_x = B[0] - A[0]  # Northing difference
    delta_y = B[1] - A[1]  # Easting difference
    distance_AB = np.sqrt(delta_x**2 + delta_y**2)
    COG = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360  # Course over ground
    
    # Extend polar data to full 0-360 degrees
    angles_port, speeds_port = polar_line
    angles_full = np.concatenate([angles_port, (360 - np.array(angles_port)) % 360])
    speeds_full = np.concatenate([speeds_port, speeds_port])

    # Sort full-circle polar data
    sorted_indices = np.argsort(angles_full)
    angles_full = angles_full[sorted_indices]
    speeds_full = speeds_full[sorted_indices]

    # Calculate VMG beat/run angles and speeds
    _ix_up = np.argmax([np.cos(np.deg2rad(a))*s for a,s in zip(angles_port, speeds_port)])
    _ix_down = np.argmin([np.cos(np.deg2rad(a))*s for a,s in zip(angles_port, speeds_port)])
    beat_angle_port, beat_speed_port = angles_port[_ix_up], speeds_port[_ix_up]
    run_angle_port, run_speed_port   = angles_port[_ix_down], speeds_port[_ix_down]
    beat_angle_starboard = 360-beat_angle_port
    run_angle_starboard  = 360-run_angle_port

    # Single leg direct to the target where possible
    direct_twa = (COG - TWD + 360) % 360  # TWA for the direct course
    port_equivalent_direct_twa = ((360 - direct_twa) if direct_twa > 180 else direct_twa) % 360 

    _idx = np.searchsorted(angles_port, port_equivalent_direct_twa, side="left") - 1
    if 0 <= _idx < len(angles_port) - 1:
        direct_sow = speeds_port[_idx]
    else:
        direct_sow = 0  # No valid direct course


    elapsed_time = distance_AB / direct_sow if direct_sow > 0 else float('inf')
    
    # VMG Method   
    if  TWA_VMC <= beat_angle_port:
        vmg_best_path = calculate_tack_or_jibe([beat_angle_port, beat_angle_starboard], [beat_speed_port, beat_speed_port], A, B, TWD)
    elif  TWA_VMC >= beat_angle_starboard:
        vmg_best_path = calculate_tack_or_jibe([ beat_angle_starboard,beat_angle_port], [beat_speed_port, beat_speed_port], A, B, TWD)
    elif run_angle_port < TWA_VMC <= 180:
        vmg_best_path = calculate_tack_or_jibe([run_angle_port,run_angle_starboard],  [run_speed_port, run_speed_port], A, B, TWD)
    elif 180 < TWA_VMC < 360 - run_angle_port:
        vmg_best_path = calculate_tack_or_jibe([run_angle_starboard,run_angle_port],  [run_speed_port, run_speed_port], A, B, TWD)
    else:
        vmg_best_path = None
    print("vmg_best_path:",vmg_best_path)
    
    # Plot the course diagram
    plt.figure(figsize=(8, 8))

    # Add the intended course AB
    plt.plot([A[1], B[1]], [A[0], B[0]], linestyle="-", color="red", label="Intended Course AB")
    plt.text((A[1] + B[1]) / 2, (A[0] + B[0]) / 2, f"Distance: {distance_AB:.2f} NM\nCOG: {COG:.2f}°\n",
             fontsize=8, color="blue")
    # Mark the points
    plt.scatter([A[1], B[1]], [A[0], B[0]], color="red", label=None) #"Points A, B")
    plt.text(A[1], A[0], "A", fontsize=12, color="red", ha="right")
    plt.text(B[1], B[0], "B", fontsize=12, color="red", ha="left")

    # Add wind direction arrow
    wind_dx = 2 * np.sin(np.radians(TWD+180))
    wind_dy = 2 * np.cos(np.radians(TWD+180))
    plt.arrow(6, 2, wind_dx, wind_dy, head_width=0.5, head_length=0.5, width=0.1, fc="green", ec="green", label="True Wind Direction")

    
    if vmc_data['leg2']:
        leg1_dist_vmc = vmc_data['leg2']["distance_leg1"]
        leg2_dist_vmc = vmc_data['leg2']["distance_leg2"]
        leg1_speed_vmc = vmc_data['leg2']['optimal_leg1_speed']
        leg2_speed_vmc = vmc_data['leg2']['optimal_leg2_speed']
        leg1_time_vmc = vmc_data['leg2']["distance_leg1"] / leg1_speed_vmc
        leg2_time_vmc = vmc_data['leg2']["distance_leg2"] / leg2_speed_vmc
        C = vmc_data['leg2']['intersection_point']
        print("VMC 2-leg via",C)
        print("vmc dist",leg1_dist_vmc+leg2_dist_vmc)
        print("vmc_time",leg1_time_vmc+leg2_time_vmc)
        plt.scatter([C[1]], [C[0]], color="magenta", label=None) #"C")
        plt.plot([A[1], C[1]], [A[0], C[0]], linestyle="--", color="magenta", label=f"vmc-leg1 {leg1_dist_vmc:.2f} Nm @ {leg1_speed_vmc:.2f} Kn, {leg1_time_vmc:.2f} Hrs")
        plt.plot([C[1],B[1]], [C[0], B[0]], linestyle="--", color="magenta", label=f"vmc-leg2 {leg2_dist_vmc:.2f} Nm @ {leg2_speed_vmc:.2f} Kn, {leg2_time_vmc:.2f} Hrs")
        
    # VMG Method
    if vmg_best_path:
        leg1_dist = vmg_best_path["distances"][0]
        leg2_dist = vmg_best_path["distances"][1]
        leg1_time = vmg_best_path["elapsedHrs"][0]
        leg2_time = vmg_best_path["elapsedHrs"][1]
        leg1_speed = vmg_best_path['speeds'][0]
        leg2_speed = vmg_best_path['speeds'][1]
        T  = vmg_best_path["tack_point"]
        print("VMG 2-leg via",T)
        print("vmg dist",leg1_dist+leg2_dist)
        print("vmg_time",leg1_time+leg2_time)
        plt.scatter([T[1]], [T[0]], color="blue", label=None) #"T")
        plt.plot([A[1], T[1]], [A[0], T[0]], linestyle="--", color="blue", label=f"vmg-leg1 {leg1_dist:.2f} Nm @ {leg1_speed:.2f} Kn, {leg1_time:.2f} Hrs")
        plt.plot([T[1],B[1]], [T[0], B[0]], linestyle="--", color="blue", label=f"vmg-leg2 {leg2_dist:.2f} Nm @ {leg2_speed:.2f} Kn, {leg2_time:.2f} Hrs")
    

    # Annotate direct course speed and elapsed time
    if direct_sow > 0:
        plt.text((A[1] + B[1]) / 2, (A[0] + B[0]) / 2,
                f"SOW: {direct_sow:.2f} kn ({elapsed_time:.2f} hrs)",
                fontsize=8, color="purple")
        print(f"Direct Course:\nSOW: {direct_sow:.2f} knots\nElapsed Time: {elapsed_time:.2f} hrs")
    plt.axis("equal")
    
    
    # plt.legend()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=10)  # Move legend below the plot
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
    num = 2*int(max(angles) - min(angles) + 1)
    smooth_angles = np.linspace(min(angles), max(angles), num)
    smooth_speeds = cubic_spline(smooth_angles)

    return smooth_angles, smooth_speeds


def plot_polar_diagram(polar_line, TWA_VMC, TWS, vmc_data, margin = 5):
    """
    Plot the polar diagram with interpolated TWS curve for both port and starboard tacks,
    VMC lines, tangent lines, and legend details. Return optimal values for use in other functions.
    """
    
    # polar_line = polars_full_rad['polar_line']
    angles_port, speeds_port = polar_line

    # Mirror port tack to get starboard tack
    angles_starboard = (360 - np.array(angles_port)) % 360  # Mirror over 0 axis
    speeds_starboard = speeds_port  # Symmetrical speeds

    # Find VMG points
    VMGs = np.array(np.array(speeds_port)) * np.cos(np.radians(np.array(angles_port)))
    _ix_up = np.argmax(VMGs)
    _ix_down = np.argmin(VMGs)
    VMG_upwind_port = angles_port[_ix_up]
    VMG_upwind_starboard = 360-angles_port[_ix_up]
    VMG_upwind_speed = speeds_port[_ix_up]
    VMG_downwind_port = angles_port[_ix_down]
    VMG_downwind_starboard = 360-angles_port[_ix_down]
    VMG_downwind_speed = speeds_port[_ix_down]

 
    # Create polar plot with increased size
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(14, 12))  # Larger figure size
    ax.set_theta_zero_location("N")  # TWA=0 at the top
    ax.set_theta_direction(-1)       # Angles increase clockwise

    # Plot port tack polar curve
    ax.plot(np.radians(angles_port), speeds_port, label=None, color="orange")

    # Plot starboard tack polar curve
    ax.plot(np.radians(angles_starboard), speeds_starboard, label=None, color="orange")
    # ax.plot(np.radians(angles_starboard), speeds_starboard, label="Starboard Tack Polar Curve", color="green")

    # Define VMC arrow parameters
    arrow_length = max(speeds_port) *1.15  # Arrow length reaches close to the perimeter
    arrow_style = f"-|>,head_length={0.1 * arrow_length},head_width={0.05 * arrow_length}"
    # Plot the VMC_TWA arrow 
    ax.annotate('', xy=(np.radians(TWA_VMC), arrow_length), xytext=(0, 0),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle=arrow_style, lw=1.5),
                )
    proxy_artist = plt.Line2D([0], [0], color='red', lw=1.5, linestyle='-', label=rf"$TWA_{{VMC}} = {{TWA_VMC:.2f}}°$")

    # Plot VMC lines and tangent lines
    def plot_vmc_and_tangent(ax, TWA_VMC, best_twa, best_sow, best_vmc, color, label_prefix):
        # VMC Line
        _q = np.cos(np.radians(best_twa - TWA_VMC))
        TWA_VMC = (TWA_VMC + 180) if _q < 0 else TWA_VMC
        _q = abs(_q)
        vmc_line_length = np.abs(best_sow * _q)
        # ax.plot([0, np.radians(TWA_VMC)], [0, best_sow], linestyle="--", color=color,
        #         label=f"{label_prefix} VMC={best_vmc:.2f} kn at TWA {TWA_VMC:.2f}°")

        # Tangent Line
        ax.plot([np.radians(best_twa), np.radians(TWA_VMC)], [best_sow, vmc_line_length],
                linestyle="--", color=color, label=rf"{label_prefix} ($TWA_{{hd}}={best_twa:.2f}°$, $VMC={best_vmc:.2f}$ kn)")

    # Port Tack
    plot_vmc_and_tangent(ax, TWA_VMC, np.degrees(vmc_data['leg2']["optimal_leg1_angle_rad"]), vmc_data['leg2']['optimal_leg1_speed'], vmc_data['leg2']['vmc1'], "magenta", "Leg1")

    # Starboard Tack
    plot_vmc_and_tangent(ax, TWA_VMC, np.degrees(vmc_data['leg2']["optimal_leg2_angle_rad"]), vmc_data['leg2']['optimal_leg2_speed'], vmc_data['leg2']['vmc2'], "magenta", "Leg2")



    # Highlight optimal VMC points
    ax.scatter([vmc_data['leg2']['optimal_leg1_angle_rad'],vmc_data['leg2']['optimal_leg2_angle_rad']],
               [vmc_data['leg2']['optimal_leg1_speed'],vmc_data['leg2']['optimal_leg2_speed']], 
               color="magenta", label= "VMC-optmized Headings") #"Optimal Point (Leg 1)")
    # ax.scatter(vmc_data['leg2']['optimal_leg2_angle_rad'],vmc_data['leg2']['optimal_leg2_speed'], color="magenta", label= None) #"Optimal Point (Lwg 2)")

    if np.cos(np.radians(TWA_VMC)) >= 0: 
        proxy_artist_vmg1 = plt.Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label=rf"$VMG = {VMG_upwind_speed * np.cos(np.radians(VMG_upwind_port))} at TWA_{{VMG}} = {VMG_upwind_port:.2f}°$")
        proxy_artist_vmg2 = plt.Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label=rf"$VMG = {VMG_upwind_speed * np.cos(np.radians(VMG_upwind_starboard))} at TWA_{{VMG}} = {VMG_upwind_starboard:.2f}°$")
    else:
        proxy_artist_vmg1 = plt.Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label=rf"$VMG = {-VMG_downwind_speed * np.cos(np.radians(VMG_downwind_port))} at TWA_{{VMG}} = {VMG_downwind_port:.2f}°$")
        proxy_artist_vmg2 = plt.Line2D([0], [0], color='blue', lw=1.5, linestyle='--', label=rf"$VMG = {-VMG_downwind_speed * np.cos(np.radians(VMG_downwind_starboard))} at TWA_{{VMG}} = {VMG_downwind_starboard:.2f}°$")

    # Mark the optimal VMG points with X
    # Plot VMG points for both port and starboard tacks
    ax.scatter(np.radians([VMG_upwind_port, VMG_upwind_starboard,VMG_downwind_port, VMG_downwind_starboard]),
            [VMG_upwind_speed, VMG_upwind_speed,VMG_downwind_speed, VMG_downwind_speed],
            marker='x', color="blue", label='VMG-optmized Beat and Run Headings')

    # ax.scatter(np.radians([VMG_downwind_port, VMG_downwind_starboard]),
    #         [VMG_downwind_speed, VMG_downwind_speed],
    #         marker='x', color="blue", label='Max Downwind VMG'),
 
    # Adjust legend placement
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, proxy_artist)
    handles.append(proxy_artist_vmg1)
    handles.append(proxy_artist_vmg2)
    labels.insert(0, rf"$TWA_{{VMC}} = {TWA_VMC:.2f}°$")
    if np.cos(np.radians(TWA_VMC)) >= 0:
        labels.append(rf"$VMG = {VMG_upwind_speed * np.cos(np.radians(VMG_upwind_port)):.2f}$ at $TWA_{{VMG}} = {VMG_upwind_port:.2f}°$")
        labels.append(rf"$VMG = {VMG_upwind_speed * np.cos(np.radians(VMG_upwind_starboard)):.2f}$ at $TWA_{{VMG}} = {VMG_upwind_starboard:.2f}°$")
    else:
        labels.append(rf"$VMG = {VMG_downwind_speed * np.cos(np.radians(VMG_downwind_port)):.2f}$ at $TWA_{{VMG}} = {VMG_downwind_port:.2f}°$")
        labels.append(rf"$VMG = {VMG_downwind_speed * np.cos(np.radians(VMG_downwind_starboard)):.2f}$ at $TWA_{{VMG}} = {VMG_downwind_starboard:.2f}°$")

    # Add the combined legend
    # ax.legend(handles=handles, labels=labels)
    ax.legend(loc="upper center", handles=handles, labels=labels, bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=10,)  # Move legend below the plot
    
    # Configure radial ticks and labels
    # ax.set_rmax(max(speeds_port) + 2)
    ax.set_rmax(max(speeds_port) * 1.15 )
    ax.set_rticks(np.arange(0, max(speeds_port) + 2, 2))
    ax.set_rlabel_position(90)

    # Tighten layout and leave space for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Add extra space at the bottom for the legend

    plt.show()

    # Return the optimal values
    # return optimal_twa_deg1, best_sow_port, best_vmc_port, best_twa_starboard, best_sow_starboard, best_vmc_starboard
 
def preprocess_polar_data(polar_line, TWD):
    """
    Prepares full polar data by including both port and starboard tacks.

    Args:
        polar_line: pair of (angles_port, speeds_port). angles_port are angles for port tack in degrees (relative to TWA).
                speeds_port (array): Corresponding speeds for port tack in knots.
        TWD (float): True Wind Direction in degrees.

    Returns:
        dict: Contains full polar data with angles, speeds, and bearings for both tacks.
    """
    
    angles_port, speeds_port = polar_line
    # Convert port angles and speeds to radians and bearings
    angles_port_rad = np.radians(angles_port)
    bearings_port = (TWD + angles_port) % 360
    bearings_port_rad = np.radians(bearings_port)

    # Generate starboard tack by mirroring port angles
    angles_starboard = (360 - np.array(angles_port)) % 360  # Mirror over 0° axis
    speeds_starboard = speeds_port  # Symmetrical speeds
    angles_starboard_rad = np.radians(angles_starboard)
    bearings_starboard = (TWD + angles_starboard) % 360
    bearings_starboard_rad = np.radians(bearings_starboard)

    # Combine port and starboard data
    angles_full_deg = np.concatenate([angles_port, angles_starboard])
    angles_full_rad = np.concatenate([angles_port_rad, angles_starboard_rad])
    speeds_full = np.concatenate([speeds_port, speeds_starboard])
    bearings_full_deg = np.concatenate([bearings_port, bearings_starboard])
    bearings_full_rad = np.concatenate([bearings_port_rad, bearings_starboard_rad])

    return {
        "angles_full_deg": angles_full_deg,
        "angles_full_rad": angles_full_rad,
        "speeds_full": speeds_full,
        "bearings_full_deg": bearings_full_deg,
        "bearings_full_rad": bearings_full_rad,
    }

def calc_vmc_results(polars_full_rad, start_point, target_point, TWA_VMC, margin=5):
    """
    Calculate VMC results for two legs and return data for optimized polar and course plots.

    Args:
        polars_full_rad (dict): Full polar data with angles, speeds, and bearings in radians.
        start_point (tuple): Starting coordinates (Northing, Easting).
        target_point (tuple): Target waypoint coordinates (Northing, Easting).
        TWA_VMC (float): True Wind Angle for VMC direction in degrees.

    Returns:
        dict: Data required for optimized polars and course plots.
    """
    # Step 1: Select optimal leg1
    leg1 = find_optimal_vmc1(polars_full_rad, TWA_VMC)


    # Step 2: Optimize leg2
    leg2 = find_optimal_vmc2(start_point, target_point, leg1, polars_full_rad, TWA_VMC,margin=margin)

    # Combine results
    return {
        "leg1": leg1,
        "leg2": leg2,
        "elapsed_time": leg2["total_elapsed_time"],
        "distances": {
            "leg1": leg2["distance_leg1"],
            "leg2": leg2["distance_leg2"],
        },
    }


def find_optimal_vmc1(polars_full_rad, TWA_VMC=0):
    """
    Find the TWA and speed on the full-circle polar curve that maximize the VMC.

    Args:
        polar_data (dict): Contains full polar data with "angles_full_rad" and "speeds_full".
        TWA_VMC (float): True Wind Angle for VMC direction in degrees.

    Returns:
        tuple: Optimal TWA (in degrees) and speed (in knots) for maximum VMC.
    """
    # Extract angles and speeds
    angles_rad = polars_full_rad["angles_full_rad"]
    speeds = polars_full_rad["speeds_full"]
    bearings_rad = polars_full_rad["bearings_full_rad"]

    # Convert TWA_VMC to radians
    TWA_VMC_rad = np.radians(TWA_VMC)

    # Compute VMC = SOW * cos(TWA - TWA_VMC)
    
    vmcs = speeds * np.cos(angles_rad - TWA_VMC_rad)

    # Find the index of the maximum VMC
    best_index = np.argmax(vmcs)

    # Extract the optimal TWA and speed
    optimal_twa_rad = angles_rad[best_index]
    optimal_twa_deg = np.degrees(optimal_twa_rad)
    optimal_speed = speeds[best_index]
    optimal_vmc  = vmcs[best_index]
    optimal_bearing_rad = bearings_rad[best_index]
    
    return {'optimal_twa_deg':optimal_twa_deg, 
            'optimal_twa_rad':optimal_twa_rad, 
            'optimal_speed':optimal_speed, 
            'optimal_vmc1':optimal_vmc,
            'optimal_bearing_rad':optimal_bearing_rad}

    
def find_optimal_vmc2(start_point, target_point, leg1, polars_full_rad, TWA_VMC, margin=5):
    """
    Calculate the optimal second leg for the VMC method.

    Args:
        start_point (tuple): Starting coordinates (Northing, Easting).
        target_point (tuple): Target waypoint coordinates (Northing, Easting).
        leg1 (dict): Optimal parameters for the first leg.
        polars_full_rad (dict): Full polar data with angles, speeds, and bearings in radians.
        TWA_VMC (float): True Wind Angle for VMC direction in degrees.

    Returns:
        dict: Optimal leg2 parameters and performance metrics, including distances and elapsed times.
    """
    # Convert TWA_VMC to radians    
    TWA_VMC_rad = np.radians(TWA_VMC)

    ixs1 = filter_polar_points_leg1(polars_full_rad["angles_full_rad"], 
                              TWA_VMC, np.radians(leg1["optimal_twa_deg"]), 
                              margin_deg=margin)
    ixs1 = np.where(ixs1)
    print(len(ixs1))

    leg1_candidates = {
        "ixs1_full":     ixs1,
        "angles_rad":   polars_full_rad["angles_full_rad"][ixs1],
        "speeds":       polars_full_rad["speeds_full"][ixs1],
        "bearings_rad": polars_full_rad["bearings_full_rad"][ixs1]
    }
    # Protect against cases where no valid candidates are found
    if len(leg1_candidates["angles_rad"]) == 0:
        return {
            "optimal_leg1_angle_rad": None,
            "optimal_leg1_speed": None,
            "intersection_point": None,
            "total_elapsed_time": None,
            "distance_leg1": None,
            "distance_leg2": None,
            "optimal_vmc1": None,
        }

    ixs2 = filter_polar_points_leg2(polars_full_rad["angles_full_rad"], 
                              TWA_VMC, np.radians(leg1["optimal_twa_deg"]), 
                              margin_deg=margin)
    ixs2 = np.where(ixs2)
    print(len(ixs2))

    leg2_candidates = {
        "ixs2_full":     ixs2,
        "angles_rad":   polars_full_rad["angles_full_rad"][ixs2],
        "speeds":       polars_full_rad["speeds_full"][ixs2],
        "bearings_rad": polars_full_rad["bearings_full_rad"][ixs2]
    }
    
    # Protect against cases where no valid candidates are found
    if len(leg2_candidates["angles_rad"]) == 0:
        return {
            "optimal_leg2_angle_rad": None,
            "optimal_leg2_speed": None,
            "intersection_point": None,
            "total_elapsed_time": None,
            "distance_leg1": None,
            "distance_leg2": None,
            # "optimal_vmc": None,
        }

    # Compute VMC for leg2 candidates
    # vmcs_leg2 = leg2_candidates["speeds"] * np.cos(leg2_candidates["angles_rad"] - TWA_VMC_rad)
    
    # Compute distances and elapsed times
    delta_x = target_point[0] - start_point[0]
    delta_y = target_point[1] - start_point[1]

    leg1_dxs = (leg1_candidates["speeds"] * np.cos(leg1_candidates["bearings_rad"])).reshape(-1,1)
    leg1_dys = (leg1_candidates["speeds"] * np.sin(leg1_candidates["bearings_rad"])).reshape(-1,1)

    leg2_dxs = (leg2_candidates["speeds"] * np.cos(leg2_candidates["bearings_rad"])).reshape(-1,1)
    leg2_dys = (leg2_candidates["speeds"] * np.sin(leg2_candidates["bearings_rad"])).reshape(-1,1)

    dets = leg1_dxs * leg2_dys.T - leg1_dys * leg2_dxs.T + 1e-8 # to aoid div by zero
    # print('det = ',det)
    t1 = (delta_x * leg2_dys - delta_y * leg2_dxs).T / dets  #if det != 0 else 0
    # t2 = (-delta_x * leg1_dy + delta_y * leg1_dx) / dets #if det != 0 else 0

    intersection_points = np.stack([start_point[0] + t1 * leg1_dxs, 
                                    start_point[1] + t1 * leg1_dys],
                                    axis=2)

    distance_leg1 = np.linalg.norm(intersection_points - np.array(start_point), axis=2)
    distance_leg2 = np.linalg.norm(np.array(target_point) - intersection_points, axis=2)
    total_elapsed_times = distance_leg1 / leg1_candidates["speeds"].reshape(-1,1) + distance_leg2 / leg2_candidates["speeds"]
    # best_idx = np.argmin(total_elapsed_times)  
    _best_idx1, _best_idx2= np.unravel_index(np.argmin(total_elapsed_times),total_elapsed_times.shape)
    # Return the result
    return {
        "optimal_leg1_angle_rad": leg1_candidates["angles_rad"][_best_idx1],
        "optimal_leg1_speed": leg1_candidates["speeds"][_best_idx1],
        "optimal_leg2_angle_rad": leg2_candidates["angles_rad"][_best_idx2],
        "optimal_leg2_speed": leg2_candidates["speeds"][_best_idx2],
        "intersection_point": intersection_points[_best_idx1,_best_idx2],
        "total_elapsed_time": total_elapsed_times[_best_idx1,_best_idx2],
        "distance_leg1": distance_leg1[_best_idx1,_best_idx2],
        "distance_leg2": distance_leg2[_best_idx1,_best_idx2,],
        "vmc1": leg1_candidates["speeds"][_best_idx1] * np.cos(leg1_candidates["angles_rad"][_best_idx1] - TWA_VMC_rad),
        "vmc2": leg2_candidates["speeds"][_best_idx2] * np.cos(leg2_candidates["angles_rad"][_best_idx2] - TWA_VMC_rad),
    }

def filter_polar_points_leg1(polar_angles, TWA_VMC, leg1_angle, margin_deg=5):
    """
    Filter polar spline points to select only the angles that:
    (a) Are "towards" the TWA_VMC (avoiding headings away from the VMC direction).
    (b) Lie on the same side of TWA_VMC as the leg1 angle.
    (c) are not further away from |TWA_VMC - leg1_angle | + small margin .

    Args:
        polar_angles (np.ndarray): Array of polar angles in radians (e.g., 1000 points from the polar spline).
        TWA_VMC (float): True Wind Angle for VMC direction in degrees.
        leg1_angle (float): Angle of the selected leg1 point in radians.
        margin_deg (float): to extend the range to improve coverage.

    Returns:
        np.ndarray: Boolean mask array of the same size as `polar_angles`.
    """
    # Convert inputs to radians and normalize
    margin_rad = np.radians(margin_deg)
    TWA_VMC_rad = np.mod(np.radians(TWA_VMC), 2 * np.pi)
    leg1_angle = np.mod(leg1_angle, 2 * np.pi)
    polar_angles = np.mod(polar_angles, 2 * np.pi)

    # condition: leg1_angle is clockwise from TWA_VMC_rad 
    _clockwise = np.mod(leg1_angle - TWA_VMC_rad, 2 * np.pi) < np.pi

    # Define the exclusion range frin TWA_VMC

    # Normalize limits to [0, 2π)

    # Create mask for angles "towards" TWA_VMC
    towards_vmc = np.cos(polar_angles - TWA_VMC_rad) > -0.25

    # Create mask for "other side" of TWA_VMC relative to leg1
    if _clockwise:
        same_side = np.sin(polar_angles - TWA_VMC_rad) >= 0
        inside_margin = polar_angles < np.mod(leg1_angle + margin_rad, 2 * np.pi)
    else:
        # Case: leg1 is counter-clockwise of TWA_VMC
        same_side = np.sin(polar_angles - TWA_VMC_rad) <= 0
        inside_margin = polar_angles > np.mod(leg1_angle - margin_rad, 2 * np.pi)

    # Combine masks
    valid_mask = towards_vmc & same_side # & inside_margin

    return valid_mask

    
def filter_polar_points_leg2(polar_angles, TWA_VMC, leg1_angle, margin_deg=2):
    """
    Filter polar spline points to select only the angles that:
    (a) Are "towards" the TWA_VMC (avoiding headings away from the VMC direction).
    (b) Lie on the other side of TWA_VMC from the leg1 angle.
    (c) Respect a margin to avoid near-zero situations.

    Args:
        polar_angles (np.ndarray): Array of polar angles in radians (e.g., 1000 points from the polar spline).
        TWA_VMC (float): True Wind Angle for VMC direction in degrees.
        leg1_angle (float): Angle of the selected leg1 point in radians.
        margin_deg (float): Margin in degrees to exclude angles too close to TWA_VMC.

    Returns:
        np.ndarray: Boolean mask array of the same size as `polar_angles`.
    """
    # Convert inputs to radians and normalize
    margin_rad = np.radians(margin_deg)
    TWA_VMC_rad = np.mod(np.radians(TWA_VMC), 2 * np.pi)
    leg1_angle = np.mod(leg1_angle, 2 * np.pi)
    polar_angles = np.mod(polar_angles, 2 * np.pi)

    # condition: leg1_angle is clockwise from TWA_VMC_rad 
    _clockwise = np.mod(leg1_angle - TWA_VMC_rad, 2 * np.pi) <= np.pi

    # Define the exclusion range around TWA_VMC
    lower_limit = TWA_VMC_rad - margin_rad
    upper_limit = TWA_VMC_rad + margin_rad

    # Normalize limits to [0, 2π)
    lower_limit = np.mod(lower_limit, 2 * np.pi)
    upper_limit = np.mod(upper_limit, 2 * np.pi)

    # Create mask for angles "towards" TWA_VMC
    towards_vmc = np.cos(polar_angles - TWA_VMC_rad) > -0.25

    # Create mask for "other side" of TWA_VMC relative to leg1
    if _clockwise:
        other_side = np.sin(polar_angles - TWA_VMC_rad) <= 0
    else:
        # Case: leg1 is counter-clockwise of TWA_VMC
        other_side = np.sin(polar_angles - TWA_VMC_rad) >= 0

    # Exclude angles within the margin around TWA_VMC
    # outside_margin = (polar_angles < lower_limit) | (polar_angles > upper_limit)

    # Combine masks
    valid_mask = towards_vmc & other_side # & outside_margin

    return valid_mask

