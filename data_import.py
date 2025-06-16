import pandas as pd
import glob

def load_streaming_data():
    """
    Loads streaming history as DataFrame.
    Returns:
        df (pd.DataFrame): Streaming history
    """
    path = "./data/Streaming_History_Audio*.json"
    file_list = glob.glob(path)
    data = []
    for file in file_list:
        print(f"Loading {file}")
        f = pd.read_json(file)
        data.append(f)
    df = pd.concat(data, ignore_index=False)
    return df