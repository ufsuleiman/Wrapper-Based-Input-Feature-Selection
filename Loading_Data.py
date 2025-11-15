import lasio
import pandas as pd

input_data=lasio.read("WLC_PETRO_COMPUTED_INPUT_1.LAS")

df=input_data.df()

df.to_excel("input_data.xlsx")
print("succesful")

output_data=lasio.read("WLC_PETRO_COMPUTED_OUTPUT_1.LAS")

df=output_data.df()

df.to_excel("input_data.xlsx")
print("succesful")


