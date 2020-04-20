import os
import csv
from datetime import datetime

os.chdir("D:/test")

year = 1901
start_month = 1
end_month = 12

for month in range(start_month, end_month+1):
	start_time = datetime.now()

	# read file
	if month < 10:
		file_name = "./" + str(year) + "0" + str(month) + ".csv"
	else:
		file_name = "./" + str(year) + str(month) + ".csv"
	
	file_reader = open(file_name, 'r', encoding="UTF-8", newline='')
	reader = csv.reader(file_reader, delimeter=",")

	file_number = 1
	count = 0

	# open new folder
	new_dir = "./test/" +str(year)
	if not os_path.exists(new_dir):
		os.makedirs(new_dir)

	# set up the first file
	file_temp = new_dir + "/" + str(year) + "_" + str(month) + "_00" + str(file_number) + ".csv"
	file_writer = open(file_temp, 'w', encoding="UTF-8", newline='')
	writer = csv.writer(file_writer)

	for arow in reader:
		writer.writerow(arow)
		count += 1

		# splite the large file to 500-row file
		if count %500 == 0:
			# print("count:", count, "row:", arow)
			file_number +=1
			file_writer.close()
			#open the next csv files
			if file_number < 10:
				file_temp =	new_dir + "/" + str(year) + "_" + str(month) + "_00" + str(file_number) + ".csv"
			elif file_number < 100:
				file_temp =	new_dir + "/" + str(year) + "_" + str(month) + "_0" + str(file_number) + ".csv"
			else:
				file_temp =	new_dir + "/" + str(year) + "_" + str(month) + "_" + str(file_number) + ".csv"
			
			file_writer = open(file_temp, 'w', encoding="UTF-8", newline='')
			writer = csv.writer(file_writer)
	
	end_time = datetime.now()
	print("Month", str(month), "is done!", "from", start_time, "to", endtime)
file_reader.close()			