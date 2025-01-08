from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import os
from pyproj import Geod

from sailutils import (
    load_boat_data,
    get_polar_data,
    plot_half_polar_data_cubicSpline,
    plot_course_diagram_with_tack,
    interpolate_polar_curve,
    plot_polar_diagram,
    preprocess_polar_data,
    calc_vmc_results
)

app_ui = ui.page_fluid(
    # Navigation bar with tabs
    ui.page_navbar(
        ui.nav_panel(
            "Boat Data Viewer",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "file", "Select a JSON file",
                        choices=[f"data/{file}" for file in os.listdir("data") if file.endswith(".json")]
                    ),
                    # ui.input_slider("TWD", "TWD", 0, 360, 72)
                ),
                ui.output_plot("polar_plot")
            )
        ),
        ui.nav_panel(
            "Course Optimization",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_numeric("A_x", "Point A Northing (NMs)", value=0),
                    ui.input_numeric("A_y", "Point A Easting (NMs)", value=0),
                    ui.input_numeric("B_x", "Point B Northing (NMs)", value=10),
                    ui.input_numeric("B_y", "Point B Easting (NMs)", value=0),
                    # ui.input_slider("TWD_opt", "True Wind Direction (°)", min=0, max=360, value=90, step=5),
                    # ui.input_slider("TWS", "True Wind Speed (knots)", min=6, max=24, value=15)
                ),
                ui.row(
                    ui.column(6,
                        ui.div(
                        ui.input_slider("TWD_opt", "", min=0, max=360, value=315, step=5),
                        ui.tags.span("TWD (°):  ", id="slider-label", style="margin-left: 5px;"),
                        style="display: flex; align-items: center;"
                        ),
                        ui.div(
                        ui.input_slider("TWS", "", min=6, max=24, value=15),
                        ui.tags.span("TWS (Kn):", id="slider-label", style="margin-left: 5px;"),
                        style="display: flex; align-items: center;"
                        ),
                        # ui.input_slider("TWD_opt", "True Wind Direction (°)", min=0, max=360, value=90, step=5),
                        ui.output_plot("course_plot", height="800px", width="100%"),),
                    ui.column(6,
                        ui.div(
                        ui.input_slider("TWA_vmc1","", min=-30, max=+30, value=0,),
                        ui.tags.span("Adjust VMC TWA 1 (°):", id="slider-label", style="margin-left: 5px;"),
                        style="display: flex; align-items: center;"
                        ),
                        ui.div(
                        ui.input_slider("TWA_vmc2","", min=-30, max=+30, value=0,),
                        ui.tags.span("Adjust VMC TWA 2 (°):", id="slider-label", style="margin-left: 5px;"),
                        style="display: flex; align-items: center;"
                        ),
                        ui.output_plot("optimized_polar_plot", height="800px", width="100%"),
                        )
                    )
                )
            ),
    ),
    # Fixed footer for the boat table
    ui.tags.div(
        ui.output_data_frame("boat_table"),
        # style="position: fixed; bottom: 0; left: 0; right: 0; height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; z-index: 10;"
        style="overflow-y: auto; height: 300px; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;"
    )
)


def server(input, output, session):
    # Reactive function to load boat data from the selected file
    @reactive.Calc
    def load_data():
        if not input.file():
            return []
        return load_boat_data(input.file())

    # Output: Display the boat data frame
    @output
    @render.data_frame
    def boat_table():
        boats = load_data()  # Load all boat data from the selected file
        if not boats:
            return pd.DataFrame()  # Return an empty DataFrame if no boats are found.

        # Flatten and simplify the boat data
        flattened_boats = [
            {
                "Seq#": i + 1,  # Add a sequence number (1-based index)
                "Sail Number": boat.get("sailnumber", ""),
                "Name": boat.get("name", ""),
                "Builder": boat.get("boat", {}).get("builder", ""),
                "Type": boat.get("boat", {}).get("type", ""),
                "Year": boat.get("boat", {}).get("year", ""),
                "Length (LOA)": boat.get("boat", {}).get("sizes", {}).get("loa", ""),
                "Beam": boat.get("boat", {}).get("sizes", {}).get("beam", ""),
                "Draft": boat.get("boat", {}).get("sizes", {}).get("draft", ""),
                "Displacement": boat.get("boat", {}).get("sizes", {}).get("displacement", ""),
            }
            for i, boat in enumerate(boats)  # Add index for Seq#
        ]

        # Convert to a pandas DataFrame
        df = pd.DataFrame(flattened_boats)

        # Render interactive DataTable with built-in filters
        return render.DataTable(
            df,
            filters=True,  # Enable column-level filtering
            selection_mode="row",  # Enable single row selection
            width="100%",  # Adjust table width as needed
        )

    @reactive.Calc
    def selected_boat():
        boats = load_data()  # Get all boats
        selected_index = boat_table.cell_selection()["rows"]  # Get selected row index
        if not selected_index or selected_index[0] >= len(boats):
            return None
        return boats[selected_index[0]]


    @output
    @render.plot
    def polar_plot():
        boat = selected_boat()
        if boat is None:
            return
        polar_data = get_polar_data(boat)
        if polar_data:
            plot_half_polar_data_cubicSpline(boat, polar_data)

    # Reactive function to calculate common variables
    @reactive.Calc
    def common_variables():
        boat = selected_boat()
        if boat is None:
            return None
        polar_data = get_polar_data(boat)
        if not polar_data:
            return None

        # A = (0,0)
        # B = (10,0)
        A = (input.A_x() or 0, input.A_y() or 0)  # Default to 0 if None
        B = (input.B_x() or 0, input.B_y() or 0)  # Default to 0 if None

        # # Ensure A and B are not identical in any dimension
        # if A[0] == B[0]:
        #     B = (B[0] + 1e-6, B[1])  # Slightly adjust longitude
        # if A[1] == B[1]:
        #     B = (B[0], B[1] + 1e-6)  # Slightly adjust latitude

        TWD = input.TWD_opt()
        TWS = input.TWS()

        # Calculate COG and TWA_VMC using pyproj's Geod
        delta_x = B[0] - A[0]  # Northing difference
        delta_y = B[1] - A[1]  # Easting difference
        distance_AB = np.sqrt(delta_x**2 + delta_y**2)
        COG = (np.degrees(np.arctan2(delta_y, delta_x)) + 360) % 360  # Course over ground

        TWA_VMC = (COG - TWD + 360) % 360
    
        # Convert distance to nautical miles
        distance_nm = distance_AB

        return {
            "boat": boat,
            "polar_data": polar_data,
            "A": A,
            "B": B,
            "TWD": TWD,
            "TWS": TWS,
            "TWA_VMC": TWA_VMC,
            "distance_nm": distance_AB,
        }

    @reactive.Calc
    def polars():
        vars = common_variables()
        polar_line = interpolate_polar_curve(vars["polar_data"], vars["TWS"])
        res = preprocess_polar_data(polar_line, vars['TWD'])
        return {"polar_line": polar_line, 
                "angles_full_rad": res["angles_full_rad"],
                'speeds_full':  res["speeds_full"],
                "bearings_full_rad": res["bearings_full_rad"]
        }

    @reactive.Calc
    def vmc_results():
        vars = common_variables()
        pol = polars()
        vmc_res = calc_vmc_results(pol, vars['A'], vars['B'], vars['TWA_VMC'])
        # Combine results
        return vmc_res


    @output
    @render.plot
    def course_plot():
        vars = common_variables()
        if not vars:
            return
        pol = polars()
        if not pol:
            return
        vmc_data = vmc_results()
        if not vmc_data:
            return

        plot_course_diagram_with_tack(
            vars["A"], vars["B"], vars["TWD"],
            pol['polar_line'],
            vars["TWA_VMC"],
            vmc_data
            # vars["distance_nm"]
        )

    @output
    @render.plot
    def optimized_polar_plot():
        vars = common_variables()
        if not vars:
            return
        pol = polars()
        if not pol:
            return
        vmc_data = vmc_results()
        if not vmc_data:
            return

        plot_polar_diagram(
            pol['polar_line'],
            vars["TWA_VMC"],
            vars["TWS"],
            vmc_data
        )


# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
