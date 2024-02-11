import glob
import json
import os
import sys

from hybrid_experiments import result_dir

filenames = glob.glob(f"{result_dir}*.json")

print("Length", len(filenames))

import sqlite3

db_name = "benchmark_results.db"
table_name = "hybrid_benchmark"
# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_name)
c = conn.cursor()

# Check if table exists
c.execute(
    f""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}' """
)
f""" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='hybrid_benchmark' """
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
            imbalances TEXT NOT NULL, 
            weight_functions TEXT NOT NULL,
            hyper_params TEXT NOT NULL,
            search_time REAL NOT NULL,
            contract_path TEXT NOT NULL,
            opt_cost REAL NOT NULL,
            time_result TEXT NOT NULL,
            trials INT
        )
    """
    )
    print("Table created successfully.")

import sqlite3

import tqdm

# Connect to SQLite database
conn = sqlite3.connect(db_name)
c = conn.cursor()

c.execute(f"DELETE FROM {table_name}")
conn.commit()
# Iterate over each JSON file
for file in tqdm.tqdm(filenames):
    # print(f"Gathering {file}")
    if os.path.exists(file):
        with open(file, "r") as f:
            data = json.load(f)

            for row in data:
                # Insert data into cotengra_benchmark table
                try:
                    c.execute(
                        """
                        INSERT INTO hybrid_benchmark (
                            circuit_file,instance,max_seconds,imbalances,weight_functions,hyper_params,search_time,contract_path,opt_cost,time_result,trials
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["circuit_file"],
                            row.get("instance", -1),
                            row["max_seconds"],
                            json.dumps(row["imbalance"]),
                            json.dumps(row["weight_function"]),
                            json.dumps(row["hyper_params"]),
                            row["search_time"],
                            json.dumps(row["contract_path"]),
                            float(row["opt_cost"]),
                            json.dumps(row["time_result"]),
                            row.get("trials"),
                        ),
                    )
                except Exception as e:
                    print(
                        (
                            row["circuit_file"],
                            row["max_seconds"],
                            json.dumps(row["imbalance"]),
                            json.dumps(row["weight_function"]),
                            json.dumps(row["hyper_params"]),
                            row["search_time"],
                            json.dumps(row["contract_path"]),
                            float(row["opt_cost"]),
                            json.dumps(row["time_result"]),
                        )
                    )
                    print(e)
                    sys.exit()
    else:
        print(f"{file} does not exist")

# Commit the changes and close the connection
conn.commit()
conn.close()
