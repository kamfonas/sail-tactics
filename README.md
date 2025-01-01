# sail-tactics

## Introduction
We compare various methods of optimizing the track from point A to point B using the VMG/VMC and polar diagrams. 
We reformatted the 2024 ORC indicative and polar plot (VPP) data from https://github.com/jieter/orc-data. We load all of them in a table at the bottom of the screen and provide a rudimentary way of filtering by various columns (sail number, name, builder, type, year, LOA, Draft, etc.). 
When a boat is selected, the top panel shows its polar plot.
A second tab enables inputs to define the coordinates Northing(x) and Easting (y) for point A and point B, the True Wind Direction (TWD) from which the wind is coming, and the True Wind Speed (TWS). On the main panel, two plots appear: (a) a course plot, including points A and B as well as other feasible two-leg tack (or jibe) routes; (b) a polar plot showing the TWS-specific curve along with a VMC-course line at the appropriate TWA corresponding to of the AB course relative to the True Wind. 
The course plot incorporates the following courses to get from A to B that are feasible:
1. A direct course, if it is possible to sail in that direction
2. A VMG course at angles that optimize the velocity-made-good along the direction of the wind for the port and starboard legs from A to the tack (or jibe) point and from there to B.
3. A VMC course optimizing the velocity-made-good towards target B, which theoretically could be better than the VMG method, but it takes the boat further away from the target, and the optimal heading changes as the boat moves.
