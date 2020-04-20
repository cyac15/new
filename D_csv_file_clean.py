import glob, os
import csv

def isNumber(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

def NewQuantity(quantity, unit_price, amount):
	if isNumber(quantity) == False:
		if (isNumber(unit_price) == True) and (float(unit_price) != 0) and (isNumber(amount) == True):
			new_value = float(amount) / float(unit_price)
			return new_value

		else:
			return ""

os.chdir("D:/test")

for file in glob.glob("*.csv"):
	count = 0
	file_origin = file
	file_reader = open(file_origin, "r", encoding="UTF-8", newline='')
	reader = csv.reader(file_reader, delimeter=",")

	# open new folder
	newdir = "new"
	if not os.path.exists(newdir):
		os.makedirs(newdir)

	file_new = "./new/" + file
	file_writer = open(file_new, "r", encoding="UTF-8", newline='')	
	writer = csv.writer(file_writer)

	for arow in reader:
		count += 1
		# 4:unit price, 6: quantity, 7: amount
		if (isNumber(arow[4]) == False and (arow[4]!= "")) or (isNumber(arow[6]) == False and (arow[6]!= "")) or (isNumber(arow[7]) == False and (arow[7]!= "")):
			new_list = []
			for i in range(0,7):
				if i == 6:
					print(arow[4], arow[7])
					row_list.append(NewQuantity(arow[6], arow[4], arow[7]))
				elif i==4 and (isNumber(arow[4]) == False):
					row_list.append("")
				elif i==7 and (isNumber(arow[7]) == False):
					row_list.append("")
				else:
					row_list.append(arow[i])
			writer.writerow(row_list)
		else:
			writer.writerow(arow)
	file_writer.close()
	file_reader.close()								
 

