import pandas as pd

# DataFrame

data = {
    "Name: ": ["Spongebob", "Patrick", "Squidward"],
    "Age: " : [30, 35, 40]
        }

df = pd.DataFrame(data)
df_index = pd.DataFrame(data, index=['Employee 1', 'Employee 2', 'Employee 3'])

print(df)
print(df_index)
print(df_index.loc["Employee 1"])
print(df_index.iloc[:])

# Adding new column

df_index["Job: "] = ['Cook', 'N/A', "Cashier"]
print(df_index.iloc[:])

# Add a new rows

new_row = pd.DataFrame([{
    "Name: ": "Sandy",
    "Age: " : 28,
    "Job: " : "Engineer"
},
{
    "Name: ": "Eugene",
    "Age: " : 16,
    "Job: " : "Manager"
}], index=["Employee 4", "Employee 5"])
df_index = pd.concat([df_index, new_row])
print(df_index)


# Homework

students = {
    "Name": ["Muhammad", 'Javohir', 'Abdulloh', 'Xurshid'],
    "Age": [32, 12, 42, 19]
}

students_df = pd.DataFrame(students, index=["Settler 1", "Settler 2", "Settler 3", "Settler 4"])
print(students_df)

# Adding new rows

students_new = pd.DataFrame([{
    "Name": 'Ali',
    "Age": 20
},
{
    "Name": 'Bunyod',
    "Age": 23
}
], index=['Settler 5', 'Settler 6'])

students_updated_list = pd.concat([students_df, students_new])
print(students_updated_list)