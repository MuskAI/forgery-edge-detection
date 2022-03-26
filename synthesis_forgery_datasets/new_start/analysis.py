"""
created by haoran
time : 20201218
data analysis tools
"""
import os
import pandas as pd
from PIL import Image
class DataAnalyse:
    def __init__(self):
        pass

    def analyse_template_size(self,template_dir):
        try:
            template_list = os.listdir(template_dir)
            row, col = [], []

            for idx, item in enumerate(template_list):
                print(idx,'/',len(template_list))
                template_path = os.path.join(template_dir, item)
                _ = Image.open(template_path)
                _size = _.size
                row.append(_size[0])
                col.append(_size[1])

            data = {'row': row, 'col': col}
            df = pd.DataFrame(data)
            writer = pd.ExcelWriter('../TempWorkShop/texture_data_size_analysis.xlsx')
            df.to_excel(writer)
            writer.save()
            print(df)
        except Exception as e:
            print(e)

if __name__ == '__main__':

    DataAnalyse().analyse_template_size(r'C:\Users\musk\Desktop\smooth5\twin_pure')