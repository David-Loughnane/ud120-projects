import pickle
import re

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


person_count = 0
salary_count = 0
total_payments_count = 0
total_stock_count = 0
poi_payment_count = 0
email_count = 0
poi_count = 0
for person in enron_data:
	person_count += 1
	poi_count += enron_data[person]['poi']
	if enron_data[person]['salary'] != 'NaN':
		salary_count += 1
	if enron_data[person]['total_payments'] != 'NaN':
		total_payments_count += 1
	if enron_data[person]['poi'] == 1 and enron_data[person]['total_payments'] != 'NaN':
		poi_payment_count += 1		
	if enron_data[person]['poi'] == 1 and enron_data[person]['total_stock_value'] != 'NaN':
		total_stock_count += 1				
	if enron_data[person]['email_address'] != 'NaN':
		email_count += 1
print('Person count: ', person_count)
print('Salary count: ', salary_count)
print('Total payments count: ', total_payments_count)
print('POIs with total payments', poi_payment_count)
print('POIs with total stock', total_stock_count)
print('Email count: ', email_count)
print('POI count: ', poi_count)


with open("../final_project/poi_names.txt", 'r') as f:
	content = f.readlines()

poi_email_counter = 0
for line in content:
	if len(line) > 1:
		if line[:3] == '(y)' or line[:3] == '(n)':
			poi_email_counter += 1
print("Inboxes of POIs: ", poi_email_counter)

print("People dictionary keys: ")
print(enron_data.keys())

print("Person dictionary keys: ")

prog = re.compile('THE TRAVEL AGENCY IN THE PARK')
for person in enron_data:
	if re.search(prog, person) != None:
		#print(person, enron_data[person]['total_payments'])
		print(person)
		for key in enron_data[person]:
			print(key, enron_data[person][key])


