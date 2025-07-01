import pandas as pd
import glob
from tqdm import tqdm

def load_streaming_data():
    """
    Loads streaming history as DataFrame.
    Returns:
        df (pd.DataFrame): Streaming history
    """
    path = "./data/in/Streaming_History_Audio*.json"
    file_list = glob.glob(path)
    data = []
    for file in tqdm(file_list, desc="Loading streaming history files"):
        f = pd.read_json(file)
        data.append(f)
    df = pd.concat(data, ignore_index=False)
    return df