import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import base64
from random import randint

### Initialise State ###
state = st.session_state

def init_state(label, default_value=None):
	if label not in state:
		state[label] = default_value


### Functions ##################################################################


def sf_ceil(x, sf=1):
    power = 10**np.floor(np.log10(abs(x))+1-sf)
    output = round(x/power)*power
    if power >= 0:
    	return int(output)
    else:
    	return output


def sf_round(x, sf=1):
	try:
		power = 10**(np.floor(np.log10(abs(x)))+1-sf)
		output = round(x/power)*power
		if power >= 0:
			return int(output)
		else:
			return output
	except ValueError:
		return x


def split_sheet(df):
    # DIVIDING ROW
    # Take the first row that has an empty value in the first column
    empty_row = df[df[0].isna()].index.tolist()[0]
    
    # VALID CELLS
    # Take each pair of columns that has a value in the pair-left column in the first data row
    # (so the first row after the Dividing Row)
    valid_cells = []
    for i in range(1, df.shape[1], 2):
        if  not np.isnan(df.loc[empty_row+1, i]):
            valid_cells.append(i)
    
    # CELL INFORMATION
    # The top section of the sheet contains the cell information
    df_info = df.iloc[0:empty_row]
    df_info.index = df_info[0]
    df_info = df_info.drop(columns=0)
    df_info = df_info[valid_cells].transpose()
    df_info["Date"] = pd.to_datetime(df_info["Date"]).dt.strftime("%m-%d")
    
    # CELL MEASUREMENTS
    # The bottom section of the sheet contains the cell measurements
    df_bottom = df.iloc[empty_row+1:].reset_index(drop=True)
    df_measurements = pd.DataFrame()
    for i in valid_cells:
        df_temp = df_bottom.loc[:, i:i+1]
        df_temp.columns = ["current", "frequency"]
        df_temp = df_temp.drop(index=df_temp.index[df_temp["current"].isna()])
        df_temp.insert(0, "cell", i)
        df_temp["normed_current"] = df_temp["current"] - df_temp["current"][0]
        df_temp = df_temp.drop(index=0)
        df_measurements = pd.concat([df_measurements, df_temp])
    
    return df_info, df_measurements


def bin_cells(df_measurements, norm=False, max_current=None, num_bins=100):
    if norm:
        col_name = "normed_current"
    else:
        col_name = "current"
        
    if max_current == None:
        max_current = sf_ceil(df_measurements[col_name].max(), 2)
        
    bins = np.histogram_bin_edges([], bins=num_bins, range=(0, max_current))
    df_binned = pd.DataFrame(columns=list(range(1, num_bins+2)))
    
    for cell_no, group in df_measurements.groupby("cell"):
        group["bin"] = np.digitize(group[col_name], bins)
        binned_data = {}
        for bin_num, g in group.groupby("bin"):
            binned_data[bin_num] = [g["frequency"].mean()]

        current_cell = pd.DataFrame(binned_data)
        current_cell.index=[cell_no]

        df_binned = pd.concat([df_binned, current_cell])

    df_binned = df_binned.astype(float).interpolate(method="slinear", axis=1)
    df_binned.columns = bins
    
    return df_binned


def get_sort_keys(s1, s2=None):
	key_dict = {}

	if s2 == None:
		for k in state["processed_data"][s1]["info"].columns.tolist()[2:]:
			key_dict[f"{k}"] = {
				"key": k,
				"sheet": s1
			}

	else:
		for s in [s1, s2]:
			for k in state["processed_data"][s]["info"].columns.tolist()[2:]:
				key_dict[f"{k} [{s}]"] = {
					"key": k,
					"sheet": s
				}

	key_dict["- None -"] = { # Set the default -None- sort value HERE.
		"key": None,
		"sheet": None
	}

	state["sort_keys"] = key_dict
	return sorted(list(key_dict.keys()))


def read_sort_key():
	key_info = state["sort_keys"][sort_key_select]
	return (key_info["key"], key_info["sheet"])


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


def generate_plot():	
	# Frequency Color Cap
	# Y-axis values - toggle (either by sort value, or off)
	# Colour Scaling

	fig = plt.figure(figsize=(6.4*plot_image_size, 4.8*plot_image_size))
	sort_key, sort_sheet = read_sort_key()

	# Single Heatmap Calculations
	if use_type == "Single Heatmap":
		dfb = bin_cells(state["processed_data"][sheet1]["measurements"],
		                norm=norm_data, max_current=plot_max_current,
		                num_bins=plot_num_bins)

		if sort_key != None:
			order = state["processed_data"][sheet1]["info"] \
					.sort_values(sort_key,
					             axis=0,
					             ascending=sort_ascending).index.tolist()

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
			                        ascending=sort_ascending)

		dfb1 = bin_cells(state["processed_data"][sheet1]["measurements"],
		                 norm=norm_data, max_current=plot_max_current,
		                 num_bins=plot_num_bins)
		# Sort sheet1 by order of sheet1_keys in combined df_info
		dfb1 = dfb1.loc[dfi[f"index{sheet1}"]].reset_index(drop=True)

		dfb2 = bin_cells(state["processed_data"][sheet2]["measurements"],
		                 norm=norm_data, max_current=plot_max_current,
		                 num_bins=plot_num_bins)
		# Sort sheet2 by order of sheet2_keys in combined df_info
		dfb2 = dfb2.loc[dfi[f"index{sheet2}"]].reset_index(drop=True)

		dfb = dfb1 - dfb2

		cmap = "vlag"
		state["plot_name"] = f"{sheet1}-{sheet2} diff sorted by {sort_key} [{sort_sheet}]"

	# Manipulate Plot
	plot = sns.heatmap(dfb, cmap=cmap)
	plot.set(title=plot_title, xlabel=plot_xlabel, ylabel=plot_ylabel)

	xticks = [sf_round(x/plot_max_current, 2) for x in range(0, int(plot_max_current)+1, int(plot_xaxis_intervals))]

	plot.set_xticks([t*plot_num_bins for t in xticks],
	                [t*plot_max_current for t in xticks])

##############################
	# CURRENT WORK IN PROGRESS
	##########################
	if plot_show_yaxis:
		current_yticks = plot.get_yticklabels()
		print(current_yticks)
		plot.set_yticks(list(range(dfb.shape[0])), 
		                [f"item_{i}" for i in range(dfb.shape[0])],
		                rotation=0)
	else:
		plot.set_yticks([])

	img = io.BytesIO()
	fig.savefig(img, format="png", bbox_inches="tight")

	state["plot_fig"] = fig
	state["plot_image"] = img


def rerender_plot():
	fig = state["plot_fig"]
	img = io.BytesIO()
	fig.savefig(img, format=download_filetype, bbox_inches="tight")
	state["plot_data"] = img
	state["plot_data_type"] = download_filetype

################################################################################

init_state("selected_file", None)
init_state("file_changed", False)
init_state("file_processed", False)
init_state("plot_fig", None)
init_state("plot_image", None)
init_state("plot_data", None)
init_state("plot_data_type", None)
init_state("upload_key", randint(1, 10**8))
init_state("reset_upload", False)

st.write("# NAME PENDING")

# Pause and wait for a file to be uploaded
if state["selected_file"] == None:
	chosen_file = st.file_uploader("Choose a file to upload:")
	if chosen_file:
		state["selected_file"] = chosen_file
		state["file_processed"] = False
		st.experimental_rerun()
	st.stop()

# If file is uploaded, process it, and then rerun the script
if not state["file_processed"]:
	loading_progress_container = st.container()
	read_excel_file(state["selected_file"], loading_progress_container)
	st.experimental_rerun()

### ONCE A FILE HAS BEEN LOADED ###

st.text(f"File selected: {state.selected_file.name}")

# Dropdown to allow selecting a new file
with st.expander("Click here to select a different file", expanded=False):
	chosen_file = st.file_uploader("Select a new file", key=state.upload_key)
if chosen_file:
	state["selected_file"] = chosen_file
	state["file_processed"] = False
	state.upload_key = randint(1, 10**8)
	st.experimental_rerun()

sheet_names = list(state.processed_data.keys())

# st.text(f"Available sheets: {sheet_names}")

cols = st.columns(2)

with cols[0]:
	use_type = st.radio("",
		                ["Single Heatmap", "Difference Heatmap"],
	    	            horizontal=True)

with cols[1]:
	st.write("# ")
	norm_data = st.checkbox("Normalise Current?",
	                        value=(use_type=="Difference Heatmap"))

data = state["processed_data"]

if use_type == "Single Heatmap":
	sheet1 = st.selectbox("Select sheet to view:", sheet_names)
	sort_keys = get_sort_keys(sheet1)

	if norm_data:
		def_max_current = sf_ceil(data[sheet1]["max_normed_current"], 2)
	else:
		def_max_current = sf_ceil(data[sheet1]["max_current"], 2)

elif use_type == "Difference Heatmap":
	diff_text = st.text(" ")
	sheet1 = st.selectbox("SHEET 1:", sheet_names)
	sheet2 = st.selectbox("SHEET 2:", [s for s in sheet_names if s != sheet1])
	diff_text.write(f"NOTE: Difference Heatmap will be ['{sheet1}' minus '{sheet2}']")
	sort_keys = get_sort_keys(sheet1, sheet2)

	if norm_data:
		def_max_current = sf_ceil(max(data[sheet1]["max_normed_current"],
		                              data[sheet2]["max_normed_current"]), 2)
	else:
		def_max_current = sf_ceil(max(data[sheet1]["max_current"],
		                              data[sheet2]["max_current"]), 2)

cols = st.columns(2)
with cols[0]:
	sort_key_select = st.selectbox("Select Sort Key:", sort_keys)

with cols[1]:
	st.write("# ")
	sort_ascending = st.checkbox("Ascending?", value=True)

with st.expander("Plot Details"):
	plot_title = st.text_input("Title:")
	plot_xlabel = st.text_input("X-Axis Label:")
	plot_ylabel = st.text_input("Y-Axis Label:")

	cols = st.columns([3,1])
	with cols[0]:
		plot_max_current = st.number_input("Max Current:", value=def_max_current)

	with cols[1]:
		st.write("# ")
		st.button("Reset")
	plot_xaxis_intervals = st.number_input("X-Axis Intervals:", 
	                                       value=sf_round(def_max_current/10),
	                                       step=100)
	plot_num_bins = st.number_input("Number of Bins (resolution):",
	                                min_value=0, max_value=1000, step=1,
	                                value=100)
	plot_image_size = st.number_input("Image Size Scale:",
	                                  min_value=0.5, max_value=100.0, step=0.1,
	                                  value=1.0)

	if read_sort_key()[0] != None:
		plot_show_yaxis = st.checkbox("Show y-axis ticks?", value=False)
	else:
		plot_show_yaxis = False



plot_button = st.button("Generate Plot", on_click=generate_plot)

if state["plot_fig"]:
	fig_plot = st.image(state["plot_image"])

	cols = st.columns(4)
	with cols[1]:
		download_filetype = st.selectbox("", ["pdf", "svg", "png", "jpg"],
		                                 on_change=rerender_plot)
	
	if state["plot_data"] == None:
		rerender_plot()

	with cols[0]:
		st.write("# ")
		download_button = st.download_button("Download",
		                                     data=state["plot_data"],
		                                     mime=f"image/{state['plot_data_type']}",
		                                     file_name=f"{state['plot_name']}.{state['plot_data_type']}")
