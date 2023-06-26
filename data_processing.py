import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import io

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