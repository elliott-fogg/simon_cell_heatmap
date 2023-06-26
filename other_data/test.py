import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def load_file():
	raw_data = pd.read_csv("MN properties summary P7-10.csv", header=None)
	data = raw_data.drop(index=raw_data.index[0:17], columns=raw_data.columns[0])
	data = data.astype(float)
	return data


def sf_ceil(x):
    return int(round(x, -int(np.floor(np.log10(abs(x))))))


NUM_BINS = 100

data = load_file()
cols = data.columns


current = []
frequency = []
for i in range(0, len(cols), 2):
    current += data.loc[:, cols[i]].tolist()
    frequency += data.loc[:, cols[i+1]].tolist()
    
max_c = sf_ceil(np.nanmax(current))
max_f = sf_ceil(np.nanmax(frequency))

bin_size = int(max_c / NUM_BINS)
print(bin_size, max_c)


@st.cache
def clean_data(data, cols, NUM_BINS):
	all_cell_data = []

	for cell_num in range(0, len(cols), 2):
	    temp = data.loc[:, cols[cell_num:cell_num+2]]
	    temp.columns = ["current", "frequency"]
	    current_cell_data = {}
	    
	    for c in range(NUM_BINS):
	        temp2 = temp[(temp.current >= c*bin_size) & (temp.current < (c+1)*bin_size)]
	        frequency = temp2.frequency.mean()
	        current_cell_data[c*bin_size] = frequency
	        
	    all_cell_data.append(current_cell_data)
	binned_data = pd.DataFrame(all_cell_data)
	binned_data = binned_data.fillna(0)

	return binned_data


binned_data = clean_data(data, cols, NUM_BINS)

fig = plt.figure(figsize=(10, 4))
sns.heatmap(binned_data)

if st.sidebar.checkbox('Show dataframe'):
	st.dataframe(binned_data)
else:
	st.text("NO SHOW")

st.pyplot(fig)