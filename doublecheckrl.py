import pickle
file_path = "q_table.pkl"

with open(file_path, 'rb') as file:
    q_table = pickle.load(file)
for i in q_table:
    print(i)