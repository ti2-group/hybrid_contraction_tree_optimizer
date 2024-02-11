import sys
import json
import os
import glob
from cotengra_experiments import (
    tasks,
    file_based_config,
    get_filename,
    result_dir,
)


filenames = glob.glob(f"{result_dir}*.json")

print("Length", len(filenames))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


import sqlite3

db_name = "benchmark_results.db"
table_name = "cotengra_benchmark"
# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_name)
c = conn.cursor()

# Check if table exists
c.execute(
    f""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}' """
)


# If the count is 1, then table exists
if c.fetchone()[0] == 1:
    print("Table exists.")
else:
    print("Table does not exist. Creating...")
    # Create table
    c.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_file TEXT NOT NULL,
            instance INT NOT NULL,
            max_seconds REAL NOT NULL,
            optlib TEXT NOT NULL,
            method TEXT NOT NULL,            
            search_time REAL NOT NULL,
            contract_path TEXT NOT NULL,
            opt_cost REAL NOT NULL,
            trials INT NOT NULL
        )
    """
    )
    print("Table created successfully.")

import sqlite3
import tqdm

# Connect to SQLite database
conn = sqlite3.connect(db_name)
c = conn.cursor()

# Clear the table
c.execute(f"DELETE FROM {table_name}")
conn.commit()
# Iterate over each JSON file
for file in tqdm.tqdm(filenames):
    # print(f"Gathering {file}")
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)
            for row in data:
                if "max_time" not in row:
                    print(file)
                    print(row["search_time"])
                    skip = (
                        len(result_dir)
                        + len(row["circuit_file"])
                        + len(str(row["instance"]))
                        + 2
                    )
                    row["max_time"] = file[skip:].split("_")[0]
                    if not is_number(row["max_time"]):
                        continue

                # Insert data into cotengra_benchmark table
                try:
                    c.execute(
                        f"""
                        INSERT INTO {table_name} (
                            circuit_file,instance,max_seconds,optlib,method,search_time,contract_path,opt_cost,trials
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)
                        """,
                        (
                            row["circuit_file"],
                            row["instance"],
                            row["max_time"],
                            row["optlib"],
                            row["method"],
                            row["search_time"],
                            json.dumps(row["contract_path"]),
                            float(row["opt_cost"]),
                            row["trials"],
                        ),
                    )
                except Exception as e:
                    print(
                        row["circuit_file"],
                        row["instance"],
                        row["max_time"],
                        row["optlib"],
                        row["method"],
                        row["search_time"],
                        json.dumps(row["contract_path"]),
                        float(row["opt_cost"]),
                        row["trials"],
                    )
                    print(e)
                    sys.exit()
    else:
        print(f"{file} does not exist")

# Commit the changes and close the connection
conn.commit()
conn.close()
