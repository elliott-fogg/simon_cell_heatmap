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

st.set_page_config("Simon's Plot Generator")

print("rerun")

### Initialise State ###
state = st.session_state

def init_state(label, default_value=None):
	global state
	if label not in state:
		state[label] = default_value

init_state("selected_file")
init_state("file_changed", False)
init_state("selected_file_name")
init_state("file_processed", False)
init_state("plot_fig")
init_state("plot_image")
init_state("plot_data")
init_state("plot_data_type", "pdf")

init_state("plot_param_defaults")
init_state("set_param_defaults", True)

init_state("set_max_current")
init_state("set_max_frequency")

init_state("plot_state")
init_state("render_state")

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

def load_sample_dataset():
	print("Trigger load sample dataset")
	r = requests.get("https://raw.githubusercontent.com/elliott-fogg/simon_cell_heatmap/main/sample_data_file.xlsx")
	state["selected_file"] = r.content
	state["file_processed"] = False
	state["selected_file_name"] = "Sample Data File"
	# st.experimental_rerun()

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
	state["file_processed"] = True
	chosen_file = None

### Plotting Functions #########################################################

def plot_details_changed():
	state["plot_state"] = None
	state["render_state"] = None
	### TODO: Put stuff in here that calculates if the current settings are the 
	#		  same as the currently plotted settings?


def start_plot():
	state["plot_state"] = "plotting"

@st.cache
def cache_bin_cells(sheet_norm, max_current, num_bins):
	dfb = bin_cells(state["processed_data"][sheet]["measurements"],
		            norm=norm, max_current=max_current, num_bins=num_bins)
	return dfb


# def single_heatmap_preprocessing():
# 	data = state["processed_data"][sheet1]["measurements"]

# 	max_current = sf_ceil(data["current"].max(), 2)
# 	max_normed_current = sf_ceil(data["normed_current"].max(), 2)
# 	max_frequency = sf_ceil(data["frequency"].max(), 2)
# 	norm = state["norm_data"]

# 	state["param_defaults"] = {
# 		"max_current": max_current,
# 		"max_normed_current": max_normed_current,
# 		"max_frequency": max_frequency,
# 		"num_bins": 100,
# 		"min_frequency": 0
# 	}

# 	dfb = cache_bin_cells(data, norm=norm, num_bins=num_bins,
# 	                      max_current=max_normed_current if norm else max_current)

# 	state["binned_data"] = dfb
# 	state["set_param_defaults"] = False


# def difference_heatmap_preprocessing():
# 	data1 = state["processed_data"][sheet1]["measurements"]
# 	data2 = state["processed_data"][sheet2]["measurements"]

# 	norm = state["norm_data"]

# 	# max_current = 












		# start = time.time()
		# dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
		#                 norm=state["norm_data"],
		#                 max_current=state["plot_max_current"],
		#                 num_bins=state["plot_num_bins"])

		# print(f"Bin Cells Time: {time.time() - start}s")

		# if sort_key != None:
		# 	dfi = state["processed_data"][sheet1]["info"] \
		# 		  .sort_values(sort_key, axis=0, 
		# 		               ascending=state["sort_ascending"])
		# 	order = dfi.index.tolist()

		# 	dfb = dfb.loc[order]
		# 	state["plot_name"] = f"{sheet1} sorted by {sort_key}"

		# else:
		# 	state["plot_name"] = f"{sheet1}"

		# cmap=None




# def single_heatmap_preprocessing():
# 	data = state["processed_data"][sheet1]["measurements"]

# 	max_current = sf_ceil(data["current"].max(), 2)
# 	max_frequency = sf_ceil(data["frequency"].max(), 2)

# 	state["param_defaults"] = {
# 		"max_current": max_current,
# 		"max_frequency": max_frequency,
# 		"num_bins": 100,
# 		"min_frequency": 0
# 	}

# 	dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
# 	                norm=state["norm_data"],
# 	                max_current=state["plot_max_current"],
# 	                num_bins=state["plot_num_bins"])








# 		if use_type == "Single Heatmap":

# 		start = time.time()
# 		dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
# 		                norm=state["norm_data"],
# 		                max_current=state["plot_max_current"],
# 		                num_bins=state["plot_num_bins"])

# 		print(f"Bin Cells Time: {time.time() - start}s")

# 		if sort_key != None:
# 			order = state["processed_data"][sheet1]["info"] \
# 					.sort_values(sort_key,
# 					             axis=0,
# 					             ascending=state["sort_ascending"]) \
# 					.index.tolist()

# 			dfb = dfb.loc[order]
# 			state["plot_name"] = f"{sheet1} sorted by {sort_key}"

# 		else:
# 			state["plot_name"] = f"{sheet1}"

# 		cmap=None


# def difference_heatmap_default_params():
# 	pass




# def difference_heatmap_preprocessing():
# 	data = state["processed_data"][sheet1]["measurements"]



# def process_single_heatmap():
# 	dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
# 	                norm=state["norm_data"], max_current=state["plot_max_current"],
# 	                num_bins=plot_num_bins)

# 	if sort_key != None:
# 		order = state["processed_data"][sheet1]["info"] \
# 			.sort_values(sort_key, axis=0, ascending=state["sort_ascending"]) \
# 			.index.tolist()
# 		dfb = dfb.loc[order]
# 		state["plot_name"] = f"{sheet1} sorted by {sort_key}"

# 	else:
# 		state["plot_name"] = sheet1

# 	cmap=state["plot_cmap"]

def calculate_difference_max():
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




def generate_plot():	
	# Frequency Color Cap / Colour Scaling

	fig = plt.figure(figsize=(6.4, 4.8))
	ax1 = fig.add_subplot(111)
	ax2 = fig.add_subplot(211)

	chosen_sort_option = state["sort_key_select"]
	_, sort_key, sort_sheet = state["sort_keys"][chosen_sort_option]

	# Single Heatmap Calculations
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

		cmap=None


	# Difference Heatmap Calculations
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

		cmap = "vlag"
		state["plot_name"] = f"{sheet1}-{sheet2} diff sorted by {sort_key} [{sort_sheet}]"

	vmax = state["plot_max_frequency"]
	vmin = state["plot_min_frequency"]

	dfb.to_csv("output_data.csv", index=False)

	# print(dfb);
	# print(dfb.max())
	# print(dfb.min())
	# print(vmin, vmax)

	cmap = "plasma"

	# Set up dual plots
	fig, (ax1, ax2) = plt.subplots(2)

	# Manipulate Plot
	sns.heatmap(ax=ax1, data=dfb, cmap=cmap, vmin=vmin, vmax=vmax)
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


	img = io.BytesIO()
	fig.savefig(img, format="png", bbox_inches="tight")

	state["plot_fig"] = fig
	state["plot_image"] = img

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

if (not state["selected_file"]) or (state["file_processed"]):
	# if not state["selected_file"]:
	# 	tabs = mySidebar.tabs(["Load data"])
	# else:
	# 	tabs = mySidebar.tabs(["Load data", "Plot Details", "Axes", "Output"])

	# with tabs[0]:
	# 	st.file_uploader("Upload a new file:", on_change=loaded_file,
	#                    	 key="file_uploader")
	# 	st.button("Use Sample Dataset", on_click=load_sample_dataset)

	cols = st.columns([2,1])

	with cols[0]:
		chosen_file = st.file_uploader("Upload a new file:",
		                               on_change=loaded_file,
		                               key="file_uploader2")

	with cols[1]:
		vert_space = '<div style="padding: 26px 0px;"></div>'
		st.markdown(vert_space, unsafe_allow_html=True)
		st.button("Use Sample Dataset", on_click=load_sample_dataset, key="1")

	if state["selected_file"] == None:

		### PLACEHOLDER FOR TESTING, REMOVE THIS WHEN DONE ###
		load_sample_dataset()
		st.experimental_rerun()
		###

		st.stop()


# If file is uploaded, process it, and then rerun the script
if not state["file_processed"]:
	loading_progress_container = st.container()
	read_excel_file(state["selected_file"], loading_progress_container)
	st.experimental_rerun()

st.text(f"File selected: {state.selected_file_name}")


### File Selection #############################################################

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

	# TODO: Get max of the combined difference heatmap

	if norm_data:
		def_max_current = sf_ceil(max(data[sheet1]["max_normed_current"],
		                              data[sheet2]["max_normed_current"]), 2)
	else:
		def_max_current = sf_ceil(max(data[sheet1]["max_current"],
		                              data[sheet2]["max_current"]), 2)

cols = st.columns(2)
with cols[0]:
	st.selectbox("Select Sort Key:", sort_options(),
	             format_func=sort_options_format_func, key="sort_key_select",
	             on_change=plot_details_changed)

with cols[1]:
	st.write("# ")
	st.checkbox("Ascending?", value=True, on_change=plot_details_changed,
	            key="sort_ascending")

st.number_input("Number of Bins (resolution):",
                min_value=10, step=1, value=100, on_change=plot_details_changed,
                key="plot_num_bins")

### Plot Details ###############################################################

with st.expander("Plot Details"):
	st.button("Reset", on_click=plot_details_changed)

	st.number_input("Max Current:",
                    value=def_max_current, on_change=plot_details_changed,
                    key="plot_max_current")

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

# if state["plot_state"] == "plotted":
# 	st.button("Generate Plot", disabled=True, type="primary")

# elif state["plot_state"] == None:
# 	st.button("Generate Plot",
# 	          on_click=start_plot,
# 	          type="primary")

if state["plot_state"] == None:
	css += """img {filter: grayscale(100%)}"""

### Plot Rendering #############################################################

if state["plot_fig"]:
	fig_plot = st.image(state["plot_image"])

	with st.expander("Plot Size Parameters"):
		st.number_input("Width (mm):", value=160, step=1,
			            on_change=render_details_changed, key="plot_width")

		st.number_input("Height (mm):", value=120, step=1,
			            on_change=render_details_changed, key="plot_height")

		st.number_input("Resolution (dpi):", value=100, step=1,
			            on_change=render_details_changed, key="plot_dpi")


	cols = st.columns([1,3])
	with cols[0]:
		st.selectbox("Download as:", ["pdf", "svg", "png", "jpg"],
		             on_change=render_details_changed, key="plot_data_type")


	if state["render_state"] == None:
		st.text("Regenerate plot to download.")

	elif state["render_state"] == "render_needed":
		st.button("Render plot for download",
		          on_click=start_render)

	elif state["render_state"] == "rendering":
		with st.spinner("Rendering..."):
			render_plot()

	elif state["render_state"] == "rendered":
		st.download_button("Download Plot",
		                   data=state["plot_data"],
		                   mime=f'image/{state["plot_data_type"]}',
		                   file_name=f'{state["plot_name"]}.{state["plot_data_type"]}')

### Final CSS Object ###########################################################
#
# Done at the end to prevent it from causing a CSS gap
#

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

################################################################################