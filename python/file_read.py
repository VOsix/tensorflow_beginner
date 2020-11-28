import sys

# path = {}
for line in sys.stdin:
    line = line.strip ()
    values = line.split ("/")
    print (values)
