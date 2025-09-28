import pandas as pd

# Series

data1 = ['a', 'b', 'c']
data2 = [100.1, 102.3, 104.3]
data3 = [True, False, True]

series1 = pd.Series(data1)
series2 = pd.Series(data2)
series3 = pd.Series(data3)

print(series1)
print(series2)
print(series3)


data = [100, 102, 104, 200, 202]
series = pd.Series(data)
print(series)

series = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])
print(series)

series.loc['c'] = 200

print(series.loc['a']) # return location
print(series.loc['b'])
print(series.loc['c'])

print(series.iloc[1]) # return location by number

print(series[series < 200])


calories = {"Day 1": 1750,
            "Day 2": 2100,
            "Day 3": 1700}

calorie_series = pd.Series(calories)

print(calorie_series)
print(calorie_series["Day 3"])

calorie_series["Day 3"] += 500
print(calorie_series["Day 3"])

print(calorie_series[calorie_series >= 2000])

# Homework

animals = ['Horse', 'Cow', 'Sheep', 'Dog', 'Cat', 'Goat', 'Hen']

animal_series = pd.Series(animals, index=['1-animal: ', '2-animal: ', '3-animal: ', '4-animal: ', '5-animal: ', '6-animal: ', '7-animal: '])
print(animal_series)

cars = {'Lovely Car: ': 'Lambo',
        'Drivable Car: ': 'Mercedes',
        'Cute Car: ': 'Porche',
        'Fast Car: ': 'F1'}
car_series = pd.Series(cars)
print(car_series)