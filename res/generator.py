import sqlite3
conn = sqlite3.connect('hackmob.db')

c = conn.cursor()

# Create table
c.execute('SELECT * FROM users WHERE cozmo_id = {}'.format(5))

user_m = c.fetchone()

print(user_m[0])

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()