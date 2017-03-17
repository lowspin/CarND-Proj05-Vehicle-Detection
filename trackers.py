class VehTracker():
    def __init__(self, nhistory):
        self.history = nhistory
        self.hot_windows = [None]*nhistory

        # self.hot_windows_0 = []  # current frame
        # self.hot_windows_1 = []  # T-1 frame
        # self.hot_windows_2 = []  # T-2 frame

    def shift_data(self):
        # self.hot_windows_2 = self.hot_windows_1
        # self.hot_windows_1 = self.hot_windows_0
        # self.hot_windows_0 = []

        for i in range(self.history-1,0,-1):
            self.hot_windows[i] = self.hot_windows[i-1]
        self.hot_windows[0] = []

    def combine_data(self):
        combined = []
        for i in range(self.history):
            if self.hot_windows[i]: # combine if list is not empty
                combined = combined + self.hot_windows[i]
        return combined

        # return self.hot_windows_0 + self.hot_windows_1 + self.hot_windows_2

    def enter_new_data(self,window_list):
        self.hot_windows[0] = window_list
        # self.hot_windows_0 = window_list
