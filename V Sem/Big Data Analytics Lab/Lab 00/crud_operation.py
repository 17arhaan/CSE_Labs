import sqlite3

connection = sqlite3.connect("aquarium.db")

cursor = connection.cursor()

# cursor.execute("CREATE TABLE IF NOT EXISTS student (name TEXT, regno INTEGER, cgpa INTEGER)")

def create_student(name, regno, cgpa):
    cursor.execute("INSERT INTO student VALUES (?, ?, ?)", (name, regno, cgpa))
    connection.commit()
    print("Student record inserted successfully.")

def read_students():
    cursor.execute("SELECT name, regno, cgpa FROM student")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

def update_student_regno(old_regno, new_regno):
    cursor.execute("UPDATE student SET regno = ? WHERE regno = ?", (new_regno, old_regno))
    connection.commit()
    print("Student record updated successfully.")

def delete_student(regno):
    cursor.execute("DELETE FROM student WHERE regno = ?", (regno,))
    connection.commit()
    print("Student record deleted successfully.")

if __name__ == "__main__":
    create_student("Arhaan", 220962050, 7)
    create_student("Arhaann", 220962051, 9)

    print("\nAll student records:")
    read_students()

    update_student_regno(220962050, 220962055)

    print("\nUpdated student records:")
    read_students()

    delete_student(220962051)

    print("\nStudent records after deletion:")
    read_students()

connection.close()