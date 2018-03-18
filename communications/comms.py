# The python half of the json test
import json
import time

def send(message, filename="tocpp"):
    print "in send"
    with open(filename, 'w') as tocpp:
        print "file opened"
        tocpp.write(message)

def recieve(filename="topy"):
    with open(filename, 'r') as topy:
        message = topy.read()
    return message

if __name__ == '__main__':
    # Emulate the autoencoder interaction
    # Read, think, write
    print "getting message"
    message = recieve()
    print "recieved ", message
    print "sleeping..."
    time.sleep(1)
    send(message)
    print "Message sent"

