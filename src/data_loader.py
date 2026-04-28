import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(base_path):
    excel_path = os.path.join(base_path, "data/cropsheet.xlsx")
    images_path = os.path.join(base_path, "data/images")

    df = pd.read_excel(excel_path)

    # Create image path column
    df['image_path'] = df['ImageName'].apply(
        lambda x: os.path.join(images_path, str(x) + ".JPG")
    )

    return df

def split_data(df):
    return train_test_split(df, test_size=0.2, random_state=42)