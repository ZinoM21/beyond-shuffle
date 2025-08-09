import glob

import pandas as pd
from tqdm import tqdm


def load_streaming_data(input_folder_name: str = "user1"):
    """
    Loads streaming history as DataFrame.
    Returns:
        df (pd.DataFrame): Streaming history
    """
    # Compose glob from provided input folder name under ./data
    path = f"./data/{input_folder_name}/Streaming_History_Audio*.json"
    file_list = glob.glob(path)
    data = []
    for file in tqdm(file_list, desc="Loading streaming history files"):
        f = pd.read_json(file)
        data.append(f)
    df = pd.concat(data, ignore_index=False)
    return df
