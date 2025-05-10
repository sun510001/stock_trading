import pandas as pd

class FixedQueueDF:
    def __init__(self, max_len=20, columns=None):
        """
        Initialize a fixed-size DataFrame queue.

        :param max_len: Maximum number of rows to keep. Default is 20.
        :param columns: List of column names for the DataFrame.
        """
        self.max_len = max_len
        self.df = pd.DataFrame(columns=columns)

    def append(self, data):
        """
        Append a new row to the end of the DataFrame queue.

        :param data: A dictionary or list/Series matching the DataFrame's column order.
        """
        new_row = pd.DataFrame([data], columns=self.df.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Trim the DataFrame to keep only the latest `max_len` rows
        if len(self.df) > self.max_len:
            self.df = self.df.iloc[-self.max_len:].reset_index(drop=True)

    def get_df(self):
        """
        Get the current DataFrame.

        :return: The internal pandas DataFrame.
        """
        return self.df
