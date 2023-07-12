import pandas as pd
import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

state = st.session_state

### Basic Functions ############################################################

def sf_ceil(x, sf=1):
    if x==0:
        return x

    power = 10**np.floor(np.log10(abs(x))+1-sf)
    output = round(x/power)*power
    if power >= 0:
        return int(output)
    else:
        return output


def sf_round(x, sf=1):
    if x == 0:
        return x

    try:
        power = 10**(np.floor(np.log10(abs(x)))+1-sf)
        output = round(x/power)*power
        if power >= 0:
            return int(output)
        else:
            return output
    except ValueError:
        return x


### Sheet Processing ###########################################################

def read_excel_file(uploaded_file, container):
    # state = st.session_state
    with container:
        temp_text = st.text("Loading datafile...")
        temp_progress = st.progress(0)

    raw_data = pd.read_excel(uploaded_file, sheet_name=None, header=None, dtype=str)

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


def split_sheet(df):
    # NOTE: Using pd.isnull() instead of np.isna(), as the latter cannot
    # handle strings.

    # DIVIDING ROW
    # Take the first row that has an empty value in the first column
    empty_row = df[df[0].isnull()].index.tolist()[0]
    
    # VALID CELLS
    # Take each pair of columns that has a value in the pair-left column in the first data row
    # (so the first row after the Dividing Row)
    valid_cells = []
    for i in range(1, df.shape[1], 2):
        if pd.notnull(df.loc[empty_row+1, i]) \
        and pd.notnull(df.loc[empty_row+1+1, i]) \
        and pd.notnull(df.loc[empty_row+1+1, i+1]):
            valid_cells.append(i)
    
    # CELL INFORMATION
    # The top section of the sheet contains the cell information
    df_info = df.iloc[0:empty_row]
    df_info.index = df_info[0]
    df_info = df_info.drop(columns=0)
    df_info = df_info[valid_cells].transpose()
    df_identity = df_info[df_info.columns[0:1]]
    df_sort = df_info[df_info.columns[1:]].astype(float)
    df_info = df_identity.join(df_sort)
    # NOTE: Don't need to convert the Date, as it is only used as an identifier?
    # df_info["Date"] = pd.to_datetime(df_info["Date"]).dt.strftime("%m-%d")
    
    # CELL MEASUREMENTS
    # The bottom section of the sheet contains the cell measurements
    df_bottom = df.iloc[empty_row+1:].reset_index(drop=True)
    df_bottom = df_bottom.astype(float)
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


@st.cache
def cache_bin_cells(sheet_name, sheet_norm, max_current, num_bins):
    print("cache - SHOULDN'T BE CALLING THIS YET")
    dfb = bin_cells(state["processed_data"][sheet_name]["measurements"],
                    norm=norm, max_current=max_current, num_bins=num_bins)
    return dfb


def generate_plot(use_type, sheet1, sheet2=None, cmap=None):
    print(f"PLOTTING: {use_type}, {sheet1}, {sheet2}")

    # Frequency Color Cap / Colour Scaling

    chosen_sort_option = state["sort_key_select"]
    _, sort_key, sort_sheet = state["sort_keys"][chosen_sort_option]

    # Single Heatmap Calculations
    if use_type == "Single Heatmap":

        print(f"Plotting Single Heatmap: {sheet1}")

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

        print(f"Plotting Difference Heatmap: {sheet1}, {sheet2}")

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

        state["plot_name"] = f"{sheet1}-{sheet2} diff sorted by {sort_key} [{sort_sheet}]"

    vmax = state["plot_max_frequency"]
    vmin = state["plot_min_frequency"]

    # dfb.to_csv("output_data.csv", index=False) # Do we need this?

    # print(dfb);
    # print(dfb.max())
    # print(dfb.min())
    # print(vmin, vmax)

    # Set up dual plots
    # fig, (ax1, ax2) = plt.subplots(2)

    fig = plt.figure(figsize=(6.4, 4.8))
    ax1 = fig.add_subplot(111)

    # Manipulate Plot
    sns.heatmap(ax=ax1, data=dfb, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set(title=state["plot_title"],
             xlabel=state["plot_xlabel"],
             ylabel=state["plot_ylabel"])


    # Set X-axis ticks
    xticks = np.arange(0, state["plot_max_current"]+1, state["plot_xaxis_intervals"])
    xticks = [x for x in range(0,
                               int(state["plot_max_current"])+1,
                               int(state["plot_xaxis_intervals"]))]

    # print("X TICKS")
    # print(xticks)
    # print(int(state["plot_max_current"]))

    # print(ax1.get_xticks())

    ax1.set_xlim(state["plot_num_bins"])

    step = 100/len(xticks)
    
    xticks2 = [t*state["plot_num_bins"]/state["plot_max_current"] for t in xticks]
    labels2 = xticks

    # print("This is the thing")
    # print(xticks2)
    # print(labels2)


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

    print(f"Saved plot for {sheet1}")

    img_pdf = io.BytesIO()
    fig.savefig(img_pdf, format="pdf", bbox_inches="tight")

    state["plot_fig"] = fig
    state["plot_image"] = img_png
    state["plot_pdf"] = img_pdf

    state["plot_state"] = "plotted"


def render_plot():
    time.sleep(2)

    mm = 1/25.4

    fig = state["plot_fig"]
    fig.set_size_inches(6.4*state["plot_width"]*mm, 4.8*state["plot_height"]*mm)
    fig.set_dpi(state["plot_dpi"])
    img = io.BytesIO()
    fig.savefig(img, format=state["plot_data_type"], bbox_inches="tight")
    state["plot_data"] = img

    state["render_state"] = "rendered"

    st.experimental_rerun()