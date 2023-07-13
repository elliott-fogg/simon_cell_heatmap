import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import requests
from file_processing import *
# from data_processing import *

### TODO: Remove all plots on reupload of a file.

### Current Set-up:
	# Generate an image as a PNG that is exactly what is shown on the screen.
	
	# Later, allow the user to generate a downloadable version that is different
	# dimensions and resolution.

	# Later, add a second axes to the figure (only on render, not shown), that
	# shows the details of the plot.

### Main Issue:
	# How do we determine when the plot parameters (e.g. Max Current,
	# Max Frequency, etc.) are reset to the current limits, and when they are
	# kept at what Simon has (presumably) set them to?

	# Could keep track of what ones he's changed, and colour-code them, and 
	# only hold the value constant if Simon has manually changed it?

	# Have a dict (?) of the max frequency and max current (for now), and log
	# both whether they've been changed, and what the custom level is.
	#
	# Have a button that is either "Fix" or "Reset", depending on their status.

### Main Issue (cont.):
	# Okay, no, the REAL issue is that we can't determine the true minimum or 
	# maximum frequency until we've binned the frequency data, as it might 
	# change depending on the Max Current or the Number of Bins (NoB).
	#
	# However, NoB is an independent variable, and the Max Current can be
	# calculated on the fly if it's not fixed.
	#
	# However, Max Frequency can be approximated from the highest Frequency
	# value in the dataset, possibly capped at the manually set Highest Current
	# (if required).
	#
	##
	#
	# This logic does NOT hold for the Difference Heatmaps though, which is 
	# going to be the truly problematic point.
	#
	# Can potentially be solved by adding a separate stage, allowing user to 
	# decide when to Bin the data?

	# First section: - Data Processing
		# Single / Difference Heatmap
		# Normalise
		# Sheet Selection
		# Number of Bins
		# Max Current

	# Second Section: - Display Settings
		# Max Frequency
		# Plot Labels
		# Axis Intervals and Ticks
		# Colourmap

	# Third Section: - Render Settings (TBA)
		# Plot Height
		# Plot Width
		# Resolution
		# Output Filetype


# Page states:
# 0 - Fresh Page
# 1 - File Selected, currently loading
# 2 - File Loaded, not processed
# 3 - File processing
# 4 - File processed, not plotted
# 5 - File plotting
# 6 - File plotted, output not generated
# 7 - Generating output
# 8 - Output generated



st.set_page_config("Simon's Plot Generator")

print("rerun")

### Initialise State ###
state = st.session_state

def init_state(label, default_value=None):
	global state
	if label not in state:
		print(f"...initiating value for {label}")
		state[label] = default_value

# File Selection Controllers
init_state("selected_file")
init_state("file_changed", False)
init_state("last_uploaded_file_id")
init_state("selected_file_name")
init_state("file_processed", False)

# Plot Data Storage
init_state("plot_fig")
init_state("plot_image")
init_state("plot_data")
init_state("plot_data_type", "pdf")

init_state("max_current_set", False)
init_state("max_current_value")
init_state("max_current_default")

init_state("max_frequency_set", False)
init_state("max_frequency_value")
init_state("max_frequency_default")

init_state("page_state", 0)

# init_state("bin_state", False)
# init_state("plot_state", False)
# init_state("render_state", False)

### Starting message in terminal output ########################################

init_state("first_run", True)

if state["first_run"]:
	print("\n\nFIRST RUN\n\n")
	state["first_run"] = False

################################################################################
#
# Functions
#
################################################################################

### Staging Functions ##########################################################

def plot_details_changed():
	### NEW
	state["page_state"] = 2
	# state["bin_state"] = False
	# state["plot_state"] = False
	# state["render_state"] = False
	###

	state["plot_state"] = None
	state["render_state"] = None
	# print(f"SHEET1: {sheet1}")


def start_binning():
	state["page_state"] = 3
	# state["bin_state"] = "binning"


def plot_details_changed():
	state["plot_state"] = None
	state["render_state"] = None
	### TODO: Put stuff in here that calculates if the current settings are the 
	#		  same as the currently plotted settings?


def start_plot():
	state["plot_state"] = "plotting"


def render_details_changed():
	state["render_state"] = "render_needed"


def start_render():
	state["render_state"] = "rendering"


### Sort Keys ##################################################################

def generate_sort_keys(s1, s2=None):
	key_list = [(None, None, s1)]

	if s2 == None:
		for k in state["processed_data"][s1]["info"].columns.tolist()[2:]:
			key_list.append( (k, k, s1) )

	else:
		for s in [s1, s2]:
			for k in state["processed_data"][s]["info"].columns.tolist()[2:]:
				key_list.append( (f"{k} [{s}]", k, s) )

	state["sort_keys"] = key_list


def sort_options():
	return list(range(len(state["sort_keys"])))


def sort_options_format_func(x):
	return state["sort_keys"][x][0]


def read_sort_key():
	chosen_sort_key = state["sort_key_select"]
	key_info = state["sort_keys"][chosen_sort_key]
	return (key_info["key"], key_info["sheet"])


### Variable Lock/Reset Functions ##############################################

def max_current_lock():
	state["max_current_value"] = state["plot_max_current"]
	state["max_current_set"] = True


def max_current_reset():
	state["max_current_set"] = False

################################################################################
#																			   #
# Actual Webpage															   #
#																			   #
################################################################################


### CSS to hide uploaded file object  ##########################################

css = """
.uploadedFile {
    display: none;
}
"""

css += """
.step-up,.step-down {
	display: none		
}
"""

### File Loading Widgets #######################################################

st.write("# Sharples' Heatmaps")
st.write(st.__version__)

load_sample_data()

# State 

with st.expander("Instructions"):
	st.download_button("Download Instructions (WIP)",
	                   data="This is a test.",
	                   mime="text/plain",
	                   file_name="simons_heatmaps_instructions.txt")
	st.download_button("Download Sample Data",
	                   data=state["sample_data"],
	                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
	                   file_name="simons_heatmaps_sample_data.xlsx")

if state["page_state"] == 1:
	loading_progress_container = st.container()
	read_excel_file(state["selected_file"], loading_progress_container)
	st.experimental_rerun()

else:
	cols = st.columns([2, 1])
	with cols[0]:
		st.file_uploader("Upload a new file:",
		                 on_change=loaded_file,
		                 key="file_uploaded")

	with cols[1]:
		vert_space = '<div style="padding: 26px 0px;"></div>'
		st.markdown(vert_space, unsafe_allow_html=True)
		st.button("Use Sample Dataset", on_click=load_sample_dataset, key="1")

if state["page_state"] < 2:
	st.stop()

print("HELP:")
print("FP: ", state["file_processed"])
print("SF: ", state["selected_file"])

# # If file is uploaded, process it, and then rerun the script
# if not state["file_processed"]:
# 	print("File not processed?")
# 	loading_progress_container = st.container()
# 	read_excel_file(state["selected_file"], loading_progress_container)
# 	st.experimental_rerun()

st.text(f"File selected: {state.selected_file_name}")


### Sheet Selection ############################################################

sheet_names = list(state.processed_data.keys())

cols = st.columns(2)
with cols[0]:
	st.radio("Mode:",
	         ["Single Heatmap", "Difference Heatmap"],
	    	 horizontal=True, label_visibility="hidden",
	    	 on_change=plot_details_changed,
	    	 key="use_type")
with cols[1]:
	st.write("# ")
	st.checkbox("Normalise Current?",
	            value=(state["use_type"]=="Difference Heatmap"),
	            on_change=plot_details_changed, key="norm_data")

data = state["processed_data"]

cols = st.columns(2)
with cols[0]:
	st.selectbox("Sheet 1:",
	             sheet_names,
	             on_change=plot_details_changed,
	             key="sheet1")


init_state("sheet1", sheet_names[0])
init_state("sheet2", sheet_names[1])

sheet1 = state["sheet1"]
use_type = state["use_type"]
sheet2 = state["sheet2"]
norm_data = state["norm_data"]

print(f"TEST - Sheet1: {sheet1}")
print(f"TEST - use_type: {use_type}")
print(f"TEST - Sheet2: {sheet2}")

if use_type == "Single Heatmap":
	generate_sort_keys(sheet1)
	if norm_data:
		def_max_current = sf_ceil(data[sheet1]["max_normed_current"], 2)
	else:
		def_max_current = sf_ceil(data[sheet1]["max_current"], 2)


elif use_type == "Difference Heatmap":
	with cols[1]:
		st.selectbox("Sheet 2:",
		             [s for s in sheet_names if s != sheet1],
		             on_change=plot_details_changed,
		             key="sheet2")
	sheet2 = state["sheet2"]

	generate_sort_keys(sheet1, sheet2)

	if norm_data:
		def_max_current = sf_ceil(max(data[sheet1]["max_normed_current"],
		                              data[sheet2]["max_normed_current"]), 2)
	else:
		def_max_current = sf_ceil(max(data[sheet1]["max_current"],
		                              data[sheet2]["max_current"]), 2)

## Sort Key and Ascending
cols = st.columns(2)
with cols[0]:
	st.selectbox("Select Sort Key:", sort_options(),
	             format_func=sort_options_format_func, key="sort_key_select",
	             on_change=plot_details_changed)
with cols[1]:
	st.write("# ")
	st.checkbox("Ascending?", value=True, on_change=plot_details_changed,
	            key="sort_ascending")

## Set 'Max Current' and 'Number of Bins'
cols = st.columns(2)
with cols[0]:
	st.number_input("Number of Bins (resolution):",
        min_value=10, step=1, value=100, on_change=plot_details_changed,
        key="plot_num_bins")


if state["max_current_set"]:
	mc_value = state["max_current_value"]
	mc_button_text = "Reset"
	mc_button_func = max_current_reset
else:
	mc_value = def_max_current
	mc_button_text = "Lock"
	mc_button_func = max_current_lock

cols = st.columns(2)
with cols[0]:
	st.number_input("Max Current:",
                    value=mc_value, on_change=plot_details_changed,
                    key="plot_max_current")
with cols[1]:
	st.write("# ")
	st.button(mc_button_text, on_click=mc_button_func)


if state["page_state"] <= 2:
	st.button("Bucket Data", on_click=start_binning)
	st.stop()

elif state["page_state"] == 3:
	with st.spinner("Plotting..."):
		process_sheets()

else:	# Page State is 4 or more
	# Show nothing, just continue the page
	pass


### Plot Details ###############################################################

with st.expander("Plot Details"):

	st.number_input("Max Frequency:", value=100, #PLACEHOLDER
	                on_change=plot_details_changed, key="plot_max_frequency")

	st.number_input("Min Frequency:", value=0, #PLACEHOLDER
	                on_change=plot_details_changed, key="plot_min_frequency")


with st.expander("Plot Axes:"):
	st.text_input("Title:", key="plot_title", on_change=plot_details_changed)

	st.text_input("X-Axis Label:", key="plot_xlabel",
	              on_change=plot_details_changed)

	st.text_input("Y-Axis Label:", key="plot_ylabel",
	              on_change=plot_details_changed)

	st.number_input("X-Axis Intervals:", value=sf_round(def_max_current/10),
	                step=100, on_change=plot_details_changed,
	                key="plot_xaxis_intervals")

	st.checkbox("Show y-axis ticks?", value=False, key="plot_show_yaxis",
	            on_change=plot_details_changed)

	st.selectbox("Colourmap:", [None, "vlag"], 
	             key="plot_cmap", on_change=plot_details_changed)


if state["plot_state"] == "plotting":
	with st.spinner("Plotting..."):
		try:
			sheet2
		except NameError:
			sheet2 = None
		generate_plot(state["use_type"], state["sheet1"], state["sheet2"])
		start_render()

st.button("Generate Plot",
          on_click=start_plot,
          type="primary",
          disabled=(state["plot_state"]=="plotted"))


if state["plot_state"] == None:
	css += """img {filter: grayscale(100%)}"""


### Plot Rendering #############################################################

if state["plot_fig"]:
	fig_plot = st.image(state["plot_image"])

	st.download_button("Download Plot",
	                   data=state["plot_pdf"],
	                   mime=f"image/pdf", # Hard-coded for now
	                   file_name=f"{state['plot_name']}.pdf");


### Final CSS Object ###########################################################
#
# Done at the end to prevent it from causing a CSS gap
#

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

################################################################################