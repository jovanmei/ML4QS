import csv

# Step 1: Read the CSV file
with open('raw_output/ch2_result_60.csv', 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Step 2: Identify the rows to delete
rows_to_delete = []
for row in rows:
    if not row['acc_phone_x'] and not row['acc_phone_y'] and not row['acc_phone_z']:
        rows_to_delete.append(row)

# Step 3: Delete the marked rows
for row in rows_to_delete:
    rows.remove(row)

# Step 4: Write the updated data to a new CSV file
fieldnames = reader.fieldnames
with open('ch2_res_60.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
