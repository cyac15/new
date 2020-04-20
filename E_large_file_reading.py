import os, glob

os.chdir("C:/Users/Desktop")

 
with open('./bus.csv', 'r', encoding='UTF-8') as in_file, open('./1.csv', 'w', encoding='UTF-8') as out_file:
    for line in in_file:
		print(line)
		out_file.write(line)
		
# can see in the excel files of Windows
with open('./bus.csv', 'r', encoding='UTF-8') as in_file, open('./1.csv', 'w', encoding='UTF-8') as out_file:
   for line in in_file:
     print(line)
     out_file.write(line)

# unicode not correct in the excel files of Windows
with open('./bus.csv', 'r', encoding='utf-8-sig') as in_file, open('./2.csv', 'w', encoding='UTF-8') as out_file:
   for line in in_file:
     print(line)
     out_file.write(line)	 
	
		
		