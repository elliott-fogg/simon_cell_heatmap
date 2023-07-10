import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import requests
from data_processing import *

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
		state[label] = default_value

# File Selection Controllers
init_state("selected_file")
init_state("file_changed", False)
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

### Loading Files ##############################################################

def load_sample_data():
	with open("sample_data_file.xlsx", "rb") as f:
		data = f.read()
		state["sample_data"] = data


def load_sample_dataset():
	print("Trigger load sample dataset")
	state["selected_file"] = state["sample_data"]
	state["selected_file_name"] = "Sample Data File"
	state["page_state"] = 1

	# # r = requests.get("https://raw.githubusercontent.com/elliott-fogg/simon_cell_heatmap/main/sample_data_file.xlsx")
	# state["selected_file"] = r.content
	# state["file_processed"] = False
	# state["selected_file_name"] = "Sample Data File"
	# # st.experimental_rerun()


def loaded_file():
	state["selected_file"] = state.file_uploader
	state["file_processed"] = False
	state["selected_file_name"] = state.file_uploader.name
	# st.experimental_rerun()

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

### Data Processing ############################################################

def read_excel_file(uploaded_file, container):
	with container:
		temp_text = st.text("Loading datafile...")
		temp_progress = st.progress(0)

	raw_data = pd.read_excel(uploaded_file, sheet_name=None, header=None)

	processed_data = {}

	sheet_names = raw_data.keys()

	for i, sheet in enumerate(sheet_names):
		temp_text.write(f"Processing sheet {i+1} / {len(sheet_names)}...")
		temp_progress.progress(i/len(sheet_names))

		df_sheet = raw_data[sheet][:]
		df_info, df_measurements = split_sheet(df_sheet)

		processed_data[sheet] = {
			"info": df_info,
			"measurements": df_measurements,
			"max_current": df_measurements["current"].max(),
			"max_normed_current": df_measurements["normed_current"].max(),
			"max_frequency": df_measurements["frequency"].max()
		}

	temp_progress.progress(1.0)
	st.session_state["processed_data"] = processed_data

	time.sleep(0.5)
	temp_progress.empty()
	temp_text.empty()
	state["page_state"] = 2
	# state["file_processed"] = True
	chosen_file = None

### Binning Functions ##########################################################

def plot_details_changed():
	state["page_state"] = 2
	# state["bin_state"] = False
	# state["plot_state"] = False
	# state["render_state"] = False


def start_binning():
	state["page_state"] = 3
	# state["bin_state"] = "binning"


@st.cache
def cache_bin_cells(sheet_norm, max_current, num_bins):
	dfb = bin_cells(state["processed_data"][sheet]["measurements"],
		            norm=norm, max_current=max_current, num_bins=num_bins)
	return dfb


def process_sheets():
	if state["max_current_set"]:
		max_current = state["max_current_value"]
	else:
		max_current = 100 # TODO: Precalculate the max current each time new sheet/s chosen

	chosen_sort_option = state["sort_key_select"]
	_, sort_key, sort_sheet = state["sort_keys"][chosen_sort_option]

	# Single Heatmap Calculations (1 sheet)
	if use_type == "Single Heatmap":

		start = time.time()
		dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
		                norm=state["norm_data"],
		                max_current=state["plot_max_current"],
		                num_bins=state["plot_num_bins"])

		print(f"Bin Cells Time: {time.time() - start}s")

		if sort_key != None:
			dfi = state["processed_data"][sheet1]["info"] \
				  .sort_values(sort_key, axis=0, 
				               ascending=state["sort_ascending"])
			order = dfi.index.tolist()

			dfb = dfb.loc[order]
			state["plot_name"] = f"{sheet1} sorted by {sort_key}"

		else:
			state["plot_name"] = f"{sheet1}"


	# Difference Heatmap Calculations (2 sheets)
	elif use_type == "Difference Heatmap":

		# Check to make sure cells match up
		dfi1 = state["processed_data"][sheet1]["info"].reset_index()
		dfi2 = state["processed_data"][sheet2]["info"].reset_index()

		dfi = dfi1.merge(dfi2, on=["Date", "Cell No"], how="inner",
		                 suffixes=(sheet1, sheet2))

		if sort_key != None:
			dfi = dfi.sort_values(f"{sort_key}{sort_sheet}", axis=0,
			                        ascending=state["sort_ascending"])

		dfb1 = bin_cells(state["processed_data"][sheet1]["measurements"],
		                 norm=state["norm_data"],
		                 max_current=state["plot_max_current"],
		                 num_bins=state["plot_num_bins"])

		# Sort sheet1 by order of sheet1_keys in combined df_info
		dfb1 = dfb1.loc[dfi[f"index{sheet1}"]].reset_index(drop=True)

		dfb2 = bin_cells(state["processed_data"][sheet2]["measurements"],
		                 norm=state["norm_data"],
		                 max_current=state["plot_max_current"],
		                 num_bins=state["plot_num_bins"])

		# Sort sheet2 by order of sheet2_keys in combined df_info
		dfb2 = dfb2.loc[dfi[f"index{sheet2}"]].reset_index(drop=True)
		dfb = dfb1 - dfb2
		state["binned_data"] = dfb

		state["plot_name"] = f"{sheet1}-{sheet2} diff sorted by {sort_key} [{sort_sheet}]"

		state["page_state"] = 4



### Variable Lock/Reset Functions ##############################################

def max_current_lock():
	state["max_current_value"] = state["plot_max_current"]
	state["max_current_set"] = True


def max_current_reset():
	state["max_current_set"] = False

### Plotting Functions #########################################################

def plot_details_changed():
	state["plot_state"] = None
	state["render_state"] = None
	### TODO: Put stuff in here that calculates if the current settings are the 
	#		  same as the currently plotted settings?


def start_plot():
	state["plot_state"] = "plotting"


def generate_plot():
	fig = plt.figure(figsize=(6.4, 4.8))
	ax1 = fig.add_subplot(111)

	# if state["max_current_set"]:
	# 	max_current = state["max_current_value"]
	# else:
	# 	max_current = 100

	# chosen_sort_option = state["sort_key_select"]
	# _, sort_key, sort_sheet = state["sort_keys"][chosen_sort_option]

	# # Single Heatmap Calculations
	# if use_type == "Single Heatmap":

	# 	start = time.time()
	# 	dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
	# 	                norm=state["norm_data"],
	# 	                max_current=state["plot_max_current"],
	# 	                num_bins=state["plot_num_bins"])

	# 	print(f"Bin Cells Time: {time.time() - start}s")

	# 	if sort_key != None:
	# 		dfi = state["processed_data"][sheet1]["info"] \
	# 			  .sort_values(sort_key, axis=0, 
	# 			               ascending=state["sort_ascending"])
	# 		order = dfi.index.tolist()

	# 		dfb = dfb.loc[order]
	# 		state["plot_name"] = f"{sheet1} sorted by {sort_key}"

	# 	else:
	# 		state["plot_name"] = f"{sheet1}"

	# 	cmap=None


	# # Difference Heatmap Calculations
	# elif use_type == "Difference Heatmap":

	# 	# Check to make sure cells match up
	# 	dfi1 = state["processed_data"][sheet1]["info"].reset_index()
	# 	dfi2 = state["processed_data"][sheet2]["info"].reset_index()

	# 	dfi = dfi1.merge(dfi2, on=["Date", "Cell No"], how="inner",
	# 	                 suffixes=(sheet1, sheet2))

	# 	if sort_key != None:
	# 		dfi = dfi.sort_values(f"{sort_key}{sort_sheet}", axis=0,
	# 		                        ascending=state["sort_ascending"])

	# 	dfb1 = bin_cells(state["processed_data"][sheet1]["measurements"],
	# 	                 norm=state["norm_data"],
	# 	                 max_current=state["plot_max_current"],
	# 	                 num_bins=state["plot_num_bins"])

	# 	# Sort sheet1 by order of sheet1_keys in combined df_info
	# 	dfb1 = dfb1.loc[dfi[f"index{sheet1}"]].reset_index(drop=True)

	# 	dfb2 = bin_cells(state["processed_data"][sheet2]["measurements"],
	# 	                 norm=state["norm_data"],
	# 	                 max_current=state["plot_max_current"],
	# 	                 num_bins=state["plot_num_bins"])

	# 	# Sort sheet2 by order of sheet2_keys in combined df_info
	# 	dfb2 = dfb2.loc[dfi[f"index{sheet2}"]].reset_index(drop=True)

	# 	dfb = dfb1 - dfb2

		# cmap = "vlag"
		

	vmax = state["plot_max_frequency"]
	vmin = state["plot_min_frequency"]

	dfb.to_csv("output_data.csv", index=False)

	print(dfb);
	print(dfb.max().max())
	print(dfb.min().min())
	print(vmin, vmax)

	# cmap = "plasma"

	# Set up dual plots
	# fig, (ax1, ax2) = plt.subplots(2)

	# Manipulate Plot
	# sns.heatmap(ax=ax1, data=dfb, cmap=cmap, vmin=vmin, vmax=vmax)
	sns.heatmap(ax=ax1, data=dfb, vmin=vmin, vmax=vmax)

	ax1.set(title=state["plot_title"],
	         xlabel=state["plot_xlabel"],
	         ylabel=state["plot_ylabel"])


	# Set X-axis ticks
	xticks = np.arange(0, state["plot_max_current"]+1, state["plot_xaxis_intervals"])
	xticks = [x	for x in range(0,
				               int(state["plot_max_current"])+1,
				               int(state["plot_xaxis_intervals"]))]

	print("X TICKS")
	print(xticks)
	print(int(state["plot_max_current"]))

	print(ax1.get_xticks())

	ax1.set_xlim(state["plot_num_bins"])

	step = 100/len(xticks)
	
	xticks2 = [t*state["plot_num_bins"]/state["plot_max_current"] for t in xticks]
	labels2 = xticks

	print("This is the thing")
	print(xticks2)
	print(labels2)


	ax1.set_xticks(xticks2, labels2)
	for _, spine in ax1.spines.items():
		spine.set_visible(True)

	### TODO #####
	# Adaptively change label size?
	# Change number of labels included on y-axis?
	##############

	if state["plot_show_yaxis"]:
		tick_positions = [x + 0.5 for x in range(dfb.shape[0])]

		if sort_key != None:
			if use_type == "Single Heatmap":


				ax1.set_yticks(tick_positions, dfi[sort_key], rotation=0)
			else:
				ax1.set_yticks(tick_positions, dfi[f"{sort_key}{sort_sheet}"], 
				                rotation=0)

		else:
			# Show Y-axis but no sort key
			ax1.set_yticks(tick_positions)

	else:
		# Don't show the Y-axis ticks
		ax1.set_yticks([])


	img_png = io.BytesIO()
	fig.savefig(img_png, format="png", bbox_inches="tight")

	img_pdf = io.BytesIO()
	fig.savefig(img_pdf, format="pdf", bbox_inches="tight")

	state["plot_fig"] = fig
	state["plot_image"] = img_png
	state["plot_pdf"] = img_pdf

	state["plot_state"] = "plotted"
	start_render()


### Rendering Plot #############################################################

def render_details_changed():
	state["render_state"] = "render_needed"

def start_render():
	state["render_state"] = "rendering"

def render_plot():
	time.sleep(5)

	mm = 1/25.4

	fig = state["plot_fig"]
	fig.set_size_inches(6.4*state["plot_width"]*mm, 4.8*state["plot_height"]*mm)
	fig.set_dpi(state["plot_dpi"])
	img = io.BytesIO()
	fig.savefig(img, format=state["plot_data_type"], bbox_inches="tight")
	state["plot_data"] = img

	state["render_state"] = "rendered"

	st.experimental_rerun()


################################################################################
#
# Actual Webpage
#
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

st.write("# COOL NAME PENDING")
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
		chosen_file = st.file_uploader("Upload a new file:",
		                               on_change=loaded_file,
		                               key="file_uploader")
	with cols[1]:
		vert_space = '<div style="padding: 26px 0px;"></div>'
		st.markdown(vert_space, unsafe_allow_html=True)
		st.button("Use Sample Dataset", on_click=load_sample_dataset, key="1")

if state["page_state"] < 2:

	### PLACEHOLDER FOR TESTING, REMOVE WHEN DONE ###
	# load_sample_dataset()
	# st.experimental_rerun()
	###

	st.stop()


# if (not state["selected_file"]) or (state["file_processed"]):

# 	cols = st.columns([2,1])

# 	with cols[0]:
# 		chosen_file = st.file_uploader("Upload a new file:",
# 		                               on_change=loaded_file,
# 		                               key="file_uploader2")

# 	with cols[1]:
# 		vert_space = '<div style="padding: 26px 0px;"></div>'
# 		st.markdown(vert_space, unsafe_allow_html=True)
# 		st.button("Use Sample Dataset", on_click=load_sample_dataset, key="1")

# 	if state["selected_file"] == None:

# 		### PLACEHOLDER FOR TESTING, REMOVE THIS WHEN DONE ###
# 		load_sample_dataset()
# 		st.experimental_rerun()
# 		###

# 		st.stop()


# # If file is uploaded, process it, and then rerun the script
# if not state["file_processed"]:
# 	loading_progress_container = st.container()
# 	read_excel_file(state["selected_file"], loading_progress_container)
# 	st.experimental_rerun()

st.text(f"File selected: {state.selected_file_name}")


### Sheet Selection ############################################################

sheet_names = list(state.processed_data.keys())

cols = st.columns(2)
with cols[0]:
	use_type = st.radio("Mode:",
		                ["Single Heatmap", "Difference Heatmap"],
	    	            horizontal=True, label_visibility="hidden",
	    	            on_change=plot_details_changed)
with cols[1]:
	st.write("# ")
	norm_data = st.checkbox("Normalise Current?",
	                        value=(use_type=="Difference Heatmap"),
	                        on_change=plot_details_changed, key="norm_data")

data = state["processed_data"]

if use_type == "Single Heatmap":
	cols = st.columns(2)
	with cols[0]:
		sheet1 = st.selectbox("Sheet 1:", sheet_names,
		                      on_change=plot_details_changed)
	
	generate_sort_keys(sheet1)

	# Check Max Current

	if norm_data:
		def_max_current = sf_ceil(data[sheet1]["max_normed_current"], 2)
	else:
		def_max_current = sf_ceil(data[sheet1]["max_current"], 2)


elif use_type == "Difference Heatmap":

	cols = st.columns(2)
	with cols[0]:
		sheet1 = st.selectbox("Sheet 1:",
		                      sheet_names,
		                      on_change=plot_details_changed)
	with cols[1]:
		sheet2 = st.selectbox("Sheet 2:",
		                      [s for s in sheet_names if s != sheet1],
		                      on_change=plot_details_changed)
	
	generate_sort_keys(sheet1, sheet2)

	# Check Max Current
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
		generate_plot()

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