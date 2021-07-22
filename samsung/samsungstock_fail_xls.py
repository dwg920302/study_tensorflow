import pandas as pd
import xlwings as xw

# data_samsung = pd.read_excel(io='./samsung/stock_samsung.xls', sheet_name='Sheet1', header=1)

# print(data_samsung)

# book = xw.Book('./samsung/stock_samsung.xls')

# book = xw.Book('./samsung/stock_samsung.xls')

book = xw.books.open('./samsung/stock_samsung.xls')

print(book)