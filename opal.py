import numpy as np
import pandas as pd


class Opal:
    def __init__(self, composition=(0.7, 0.28, 0.02)):
        self.X, self.Y, self.Z = composition
        # find correct table number
        self.summary = self.load_opacity_summary("GN93HZ")
        num = 1
        for tb in self.summary:
            if float(tb[1]) == self.X and float(tb[2]) == self.Y and float(tb[3]) == self.Z:
                num = int(tb[0])
        self.df = self.read_opacity_table("GN93HZ", num)

    # load summaries of tables to find table with the correct composition
    def load_opacity_summary(self, fn, skip_header=62, max_rows=126, usecols=(1, 2, 4, 5, 6, 7)):
        # format: [Table no., X, Y, Z]
        summary = np.genfromtxt(fn, dtype='str', skip_header=skip_header,
                                max_rows=max_rows, usecols=usecols, comments=None)

        # get table num, X, Y and Z columns
        for i, row in enumerate(summary):
            if i > 98:
                row[0] = row[0][1:]
                row[1] = row[2][2:]
                row[2] = row[3][2:]
                row[3] = row[4][2:]
            else:
                row[0] = row[1][0:]
                row[1] = row[3][2:]
                row[2] = row[4][2:]
                row[3] = row[5][2:]
        
        summary = summary[:, :4]
        # cast first column (table nums) to integer
        summary[:, 0] = summary[:, 0].astype(int)
        # cast last 3 columns (compositions) to float
        summary[:, 1:] = summary[:, 1:].astype(float)
        return summary

    # reads in opacity table # table_num
    def read_opacity_table(self, fn, table_num):
        table_size = 70
        table_1_start = 245
        table_row = table_1_start + (table_num-1)*(table_size+7)
        table = pd.read_table(fn, dtype=float, header=0, index_col=0,
                              skiprows=table_row, nrows=table_size, sep="\s+", na_values=9.999)
        return table

    # returns the indices before and after the position of x in the sorted array arr
    def find_closest_indices(self, arr, x):
        index = np.argmax(arr >= x)
        return index - 1, index

    def bilinear_interp(self, x, y, x1, y1, x2, y2, Q11, Q12, Q21, Q22):
        R1 = Q11 * ((x2 - x) / (x2 - x1)) + Q21 * ((x - x1) / (x2 - x1))
        R2 = Q12 * ((x2 - x) / (x2 - x1)) + Q22 * ((x - x1) / (x2 - x1))

        return R1 * ((y2 - y) / (y2 - y1)) + R2 * ((y - y1) / (y2 - y1))

    def get_opacity(self, rho, T):
        T6 = 1e-6 * T
        R = rho / (T6 ** 3)
        log_T = list(self.df.index)
        log_R = [float(x) for x in self.df.columns]

        i1, i2 = self.find_closest_indices(log_T, np.log10(T))
        j1, j2 = self.find_closest_indices(log_R, np.log10(R))
        x1, x2 = log_T[i1], log_T[i2]
        y1, y2 = log_R[j1], log_R[j2]
        Q11, Q12, Q21, Q22 = self.df.iloc[i1, j1], self.df.iloc[i1,
                                                                j2], self.df.iloc[i2, j1], self.df.iloc[i2, j2]
        log_k = self.bilinear_interp(x=np.log10(T), y=np.log10(
            R), x1=x1, y1=y1, x2=x2, y2=y2, Q11=Q11, Q12=Q12, Q21=Q21, Q22=Q22)

        return 10 ** log_k
