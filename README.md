# sail-tactics

Remember that this project is under development and incomplete; forgive our untidy code. 

## Introduction
We compare various methods of optimizing the track from point A to point B using the VMG/VMC and polar diagrams. 
We reformatted the 2024 ORC indicative and polar plot (VPP) data from https://github.com/jieter/orc-data. We load all of them in a table at the bottom of the screen and provide a rudimentary way of filtering by various columns (sail number, name, builder, type, year, LOA, Draft, etc.). This list of over 5000 boats is scrollable and is always visible at the bottom of the window (you may have to scroll down a little). When the user selects a boat, that boat is used in the upper panel of the window.
The first navigation tab, **Boat Data Viewer** shows the selected boat's indicative data and polar plot.
The second tab, **Course Optimization**, allows inputs of the sidebar to control some of the static parameters, such as the Northing(x) and Easting (y) for point A and point B of the intended course, and parameters that affect the algorithms used. The True Wind Direction (TWD) from which the wind is coming and the True Wind Speed (TWS) are controlled from the top sliders of the main panel, which shows two plots: (a) A course plot, including points A and B, the distance and COG bearing, as well as speed over water (SOW) and elapsed time when the direct route is feasible. Two-leg tack or jibe routes are also shown if possible; (b) A polar plot slice with the TWS-specific curve, the VMC-course line at the appropriate true wind angle (TWA) corresponding to the AB course relative to the True Wind, and VMG-optimized headings.
More specifically, the following optimization methods for the course A to B are shown when feasible:
1. A direct route, if it is possible to sail in that direction, extending the course annotation to include SOW and Elapsed time in magenta color.
2. A VMG route at angles that optimize the velocity-made-good along the wind direction (VMG) for the two legs from A to the tack (or jibe) point and from there to B. Speeds and elapsed time are shown in the legend for each leg.
3. A VMC static route, optimizing the velocity-made-good towards target B (VMC), which theoretically can be better than the VMG method. However, it takes the boat further away from the target, and after a short distance along leg 1, the optimal heading shifts until it overlaps with the VMG heading. Since the VMC implementation is static and leg1 is a straight line whose angle is determined at point A, by the end of leg1, the TWA is no longer optimal.
4. A VMC adaptive course (under development) addresses the issue described in 3 above. This approach builds the route in small consecutive increments optimized based on the latest boat position until it converges with the VMG-optimized TWA, and it reverts to a VMG method for the rest of leg1.  
5. A dynamically optimized route (under development) uses intelligently selected pairs of angles for the two legs, and combines random exploration of new values with exploitation of knowledge obtained in previous trials to converge to optimal angles that minimize the total time.

Items 1-3 have been implemented as of now. We currently assume fixed wind direction and speed. We also ignore tides and leeway drag effects. We plan to extend to those areas later. 

## Application Access ##

A dev version of the app can be found [here](https://www.shinyapps.io/admin/#/application/13662471). It is periodically updated after corrections or new capabilities are added. 


