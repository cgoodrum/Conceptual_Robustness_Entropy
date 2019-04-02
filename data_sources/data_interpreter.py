import csv
import numpy as np
import random as rd

class Case_Data(object):
    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def __init__(self, filename = ''):
        self.filename = filename
        self.data = self.read_csv_file()

    def read_csv_file(self):
        data = {}
        with open(self.filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                data[line_count] = {
                    "Ship": str(row["Oceanograpic Ships"]),
                    "L": float(row["L"]),
                    "B": float(row["B"]),
                    "T": float(row["T"]),
                    "Cb": float(row["Cb"])
                }
                line_count += 1
            return data

    def calc_vols(self):
        vol_dict = {}
        for k, v in self.data.items():
            vol_dict[k] = {
                "V": float(v['L']*v['B']*v['T']*v['Cb'])
            }
        return vol_dict

    def calc_avg_vol_from_vols(self):
        vols = self.calc_vols()
        avg_vol = np.mean(vols.values())
        return avg_vol

    def calc_var_avgs(self):
        L_dict = {}
        B_dict = {}
        T_dict = {}
        Cb_dict = {}
        avgs_dict = {}
        for k1, v1 in self.data.items():
                L_dict[k1] = v1['L']
                B_dict[k1] = v1['B']
                T_dict[k1] = v1['T']
                Cb_dict[k1] = v1['Cb']
        L_avg = np.mean(L_dict.values())
        B_avg = np.mean(B_dict.values())
        T_avg = np.mean(T_dict.values())
        Cb_avg = np.mean(Cb_dict.values())
        avgs_dict = {
        "L": L_avg,
        "B": B_avg,
        "T": T_avg,
        "Cb": Cb_avg,
        }
        return avgs_dict

    def calc_avg_vol_from_vars(self):
        avg = self.calc_var_avgs()
        avg_vol = avg['L']*avg['B']*avg['T']*avg['Cb']
        return avg_vol

    def randomize(self):
        random_data = {}

        count = 1
        for key, value in sorted(self.data.items(), key=lambda x: rd.random()):
            #print key, value
            random_data[count] = value
            count +=1

        self.data = random_data
        #return random_data
