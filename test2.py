import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if "file_selected" not in st.session_state:
	st.session_state["file_selected"] = None
if "file_changed" not in st.session_state:
	st.session_state["file_changed"] = False

if st.session_state["file_selected"] == None:
	file_selected = False
else:
	file_selected = True
	file_uploaded = st.session_state["file_selected"]
	if ("raw_data" not in st.session_state) or st.session_state["file_changed"]:
		st.session_state["raw_data"] = pd.read_excel(file_uploaded,
	                                             	 sheet_name=None,
	                                             	 header=None)

if not file_selected:
	chosen_file = st.file_uploader("Choose a file to upload:")
	if chosen_file:
		st.session_state["file_selected"] = chosen_file
		st.write("Loading file...")
		st.experimental_rerun()
	st.stop()

st.text(f"File selected: {file_uploaded.name}")
with st.expander("Click here to select a different file"):
	chosen_file = st.file_uploader("Select a new file")
if chosen_file:
	st.session_state["file_selected"] = chosen_file
	st.experimental_rerun()

sheet_names = list(st.session_state["raw_data"].keys())

st.text(f"Available sheets: {sheet_names}")

use_type = st.radio("Select use type:", ["Plot Single", "Plot Comparison",
                    					 "Examine Raw Data",
                    					 "Examine Processed Data"],
                    horizontal=True)

if use_type == "Plot Single":
	sheet1 = st.selectbox("Select sheet to view:", sheet_names)

elif use_type == "Plot Comparison":
	sheet1 = st.selectbox("Select first sheet:", sheet_names)
	sheet2 = st.selectbox("Select second sheet:", sheet_names)

elif use_type == "Examine Raw Data":
	sheet1 = st.selectbox("Select sheet to view:", sheet_names)
	st.dataframe(st.session_state["raw_data"][sheet1])

elif use_type == "Examine Processed Data":
	st.text("WIP")

st.button("GENERATE")

# st.download_button(label, data, file_name=None, mime=None, key=None, help=None,
# on_click=None, args=None, kwargs=None, *, disabled=False)

# st.slider(label, min_value=None, max_value=None, value=None, step=None,
# format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *,
# disabled=False, label_visibility="visible")

# st.radio(label, options, index=0, format_func=special_internal_function,
# key=None, help=None, on_change=None, args=None, kwargs=None, *,
# disabled=False, horizontal=False, label_visibility="visible")

st.write("BOTTOM")