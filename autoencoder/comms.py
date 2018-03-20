# The python half of the json test
import json
import time

def send(message, filename="tocpp"):
    #print "in send"
    with open(filename, 'w') as tocpp:
        #print "file opened"
        tocpp.write(message)
        tocpp.flush()

def recieve(filename="topy"):
    with open(filename, 'r') as topy:
        message = topy.read()
    return message

if __name__ == '__main__':
    # Emulate the autoencoder interaction
    # Read, think, write
    #print "getting messages"
    message = recieve()
    message = recieve()
    #print "recieved ", message
    send(message)
    #print "Message sent"

