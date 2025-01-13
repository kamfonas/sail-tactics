import numpy as np
from skopt import gp_minimize
from skopt.space import Real

class VMCOptim:
    def __init__(self, polar_data, course_angle, target_distance):
        """
        Initializes the VMCOptim class.

        Args:
            polar_data (dict): Polar data dictionary with wind angles and boat speeds.
            course_angle (float): The course angle from start to waypoint in degrees.
            target_distance (float): Distance from start to waypoint in nautical miles.
        """
        self.polar_data = polar_data
        self.course_angle = course_angle
        self.target_distance = target_distance
        self.result = None

    def score(self, params):
        """
        Scoring function for elapsed time based on leg angles.

        Args:
            params (list): List of parameters [leg1_angle, leg2_angle].

        Returns:
            float: Negative of total elapsed time.
        """
        leg1_angle, leg2_angle = params

        # Compute speeds using polar data
        speed_leg1 = self.get_speed(leg1_angle)
        speed_leg2 = self.get_speed(leg2_angle)

        # Compute distances for legs
        distance_leg1 = self.target_distance * np.cos(np.radians(leg1_angle))
        distance_leg2 = self.target_distance * np.sin(np.radians(leg1_angle))

        # Compute times for each leg
        time_leg1 = distance_leg1 / speed_leg1 if speed_leg1 > 0 else float('inf')
        time_leg2 = distance_leg2 / speed_leg2 if speed_leg2 > 0 else float('inf')

        total_time = time_leg1 + time_leg2

        return -total_time  # Negative because we minimize

    def get_speed(self, angle):
        """
        Interpolates the boat speed from the polar data for a given angle.

        Args:
            angle (float): The angle (TWA) in degrees.

        Returns:
            float: Interpolated boat speed.
        """
        angles = np.array(list(self.polar_data.keys()))
        speeds = np.array(list(self.polar_data.values()))
        return np.interp(angle, angles, speeds)

    def setup_space(self):
        """
        Sets up the search space for leg angles.

        Returns:
            list: List of search space dimensions for gp_minimize.
        """
        return [
            Real(0, 90, name='leg1_angle'),  # Leg1 angle from 0 to 90 degrees
            Real(self.course_angle, self.course_angle + 90, name='leg2_angle')  # Leg2 angle relative to the course
        ]

    def run_optimization(self, n_calls=30):
        """
        Runs the gp_minimize optimization to find the optimal leg angles.

        Args:
            n_calls (int): Number of iterations for the optimization.

        Returns:
            dict: Result dictionary containing optimal angles and elapsed time.
        """
        space = self.setup_space()
        self.result = gp_minimize(self.score, space, n_calls=n_calls, acq_func='EI')

        return {
            'optimal_leg1_angle': self.result.x[0],
            'optimal_leg2_angle': self.result.x[1],
            'minimum_time': -self.result.fun
        }

    def evaluate(self):
        """
        Evaluates and generates structures for polar plots and course diagrams.

        Returns:
            dict: Evaluation results containing optimal data for visualization.
        """
        if self.result is None:
            raise ValueError("Optimization has not been run. Call run_optimization first.")

        return {
            'polar_plot': {
                'angles': self.result.x,
                'speeds': [
                    self.get_speed(self.result.x[0]),
                    self.get_speed(self.result.x[1])
                ]
            },
            'course_diagram': {
                'leg1': {
                    'angle': self.result.x[0],
                    'speed': self.get_speed(self.result.x[0])
                },
                'leg2': {
                    'angle': self.result.x[1],
                    'speed': self.get_speed(self.result.x[1])
                }
            }
        }
