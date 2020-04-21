import pandas as pd


def load_mode_data(filename):

    mode_df = pd.read_csv(filename, header=0, delim_whitespace=True)
    star_id = mode_df['id'].values[0]
    mode_deg = mode_df['l'].values
    mode_freq = mode_df['freq'].values
    mode_freq_err = mode_df['freq_err'].values

    return star_id, (mode_freq[mode_deg == 0], mode_freq[mode_deg == 1], mode_freq[mode_deg == 2])



    
