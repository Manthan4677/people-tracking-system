import math
from typing import List
import psycopg2
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(override=True)

conn = psycopg2.connect(os.environ["POSTGRES_URI"])


def create_csv(conn):
    curr = conn.cursor()
    sql = "COPY (SELECT MIN(time) as first_seen , MAX(time) as last_seen ,section,id from visit_times group by section , id) TO STDOUT WITH CSV DELIMITER ',' HEADER"
    with open("./table.csv", "w") as file:
        curr.copy_expert(sql, file)
    data = pd.read_csv("table.csv")
    print(data)


def plot_person_activity(conn, id):
    curr = conn.cursor()
    sql = "SELECT time, section from visit_times WHERE id = %s ORDER BY time ASC"
    curr.execute(sql, (id,))
    rows = curr.fetchall()
    df = pd.DataFrame(rows, columns=["time", "section"])

    fig, ax = plt.subplots()
    ax.plot(df["time"], df["section"], label=id)
    # plt.xticks(df['time'].values)
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz="Asia/Kolkata"))
    fig.autofmt_xdate()
    fig.legend()
    plt.show()


def plot_barchart(conn, sections: List[str]):
    X = [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "> 70",
    ]
    curr = conn.cursor()
    cols = 2
    fig, ax = plt.subplots(
        ncols=cols, nrows=math.ceil(len(sections) / cols), figsize=(8, 8)
    )

    for j, section in enumerate(sections):

        sql = """
        WITH cte AS (
            SELECT MAX(time) - MIN(time) as time_spent, v.gender, v.age, vt.section FROM visit_times vt INNER JOIN visitors v ON (vt.id = v.id) GROUP BY vt.id, v.gender, v.age, vt.section
        )
        SELECT time_spent, gender, age FROM cte WHERE section = %s ORDER BY age ASC;
        """
        curr.execute(sql, (section,))
        rows = curr.fetchall()

        maleY = [0] * len(X)
        femaleY = [0] * len(X)

        males = [row for row in rows if row[1] == "male"]
        females = [row for row in rows if row[1] == "female"]

        for i, x in enumerate(X):
            for male in males:
                if male[2] == x:
                    maleY[i] = male[0].total_seconds()
                    break

            for female in females:
                if female[2] == x:
                    femaleY[i] = female[0].total_seconds()
                    break

        X_axis = np.arange(len(X))

        row = j // cols
        col = j % cols

        ax[row, col].bar(X_axis - 0.2, maleY, 0.4, label="Males")
        ax[row, col].bar(X_axis + 0.2, femaleY, 0.4, label="Females")
        ax[row, col].set_xticks(X_axis, minor=False)
        ax[row, col].set_xticklabels(X, fontdict=None, minor=False, rotation=25)
        ax[row, col].set_title(section)

        ax[row, col].set_xlabel("Age Groups")
        ax[row, col].set_ylabel("Time spent")
        ax[row, col].legend()
    if len(sections) % 2 != 0:
        for i in range(len(sections) % cols, cols):
            fig.delaxes(ax[len(sections) // cols, i])

    fig.tight_layout()
    fig.suptitle("Customer distribution by gender and age group")
    plt.show()


def select_person_id(conn):
    curr = conn.cursor()
    curr.execute("SELECT id FROM visitors")
    rows = curr.fetchall()

    print("Select person: ")
    for i, row in enumerate(rows):
        print(f"[{i:2d}] {row[0]}")

    choice = int(input("your choice: "))
    return rows[choice][0]


if __name__ == "__main__":
    print("1. Linegraph \n2. Bargraph\n3. Create Csv\n")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        person_id = select_person_id(conn)
        plot_person_activity(conn, person_id)
    elif choice == 2:
        plot_barchart(conn, ["child", "male", "female", "main", "cashier"])
    elif choice == 3:
        create_csv(conn)
    else:
        print("Not a Valid Choice")
        exit(0)
