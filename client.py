  
#CLIENT
import asyncio
import websockets
import time
from time import sleep
 
# Raspberry Pi specific Libraries & Variables
'''
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
 
Ena, In1, In2 = 17, 27, 22
 
GPIO.setup(Ena, GPIO.OUT)
GPIO.setup(In1, GPIO.OUT)
GPIO.setup(In2, GPIO.OUT)
 
pwm = GPIO.PWM(Ena, 100)
pwm.start(0)
'''
 
 
async def run_client():
    async with websockets.connect("ws://localhost:8765") as websocket:
        #First Message sent when connection is established
        message = "Connection Establised!"
        await websocket.send(message)
        while True:
            time.sleep(0.005) #for debuggging purposes
            direction = await websocket.recv()
            print("Received direction from server: {}".format(direction))
 
            if direction == "FORWARD":
                #Give both motors forward high
                # GPIO.output(In1, GPIO.HIGH)
                # GPIO.output(In2, GPIO.HIGH)
                print("Direction: {} - FORWARD".format(direction))
            if direction == "BACKWARD":
                #Give both motors forward high
                # GPIO.output(In1, GPIO.HIGH)
                # GPIO.output(In2, GPIO.HIGH)
                print("Direction: {} - BACKWARD".format(direction))
            if direction == "LEFT":
                #Give both motors forward high
                # GPIO.output(In1, GPIO.LOW)
                # GPIO.output(In2, GPIO.HIGH)
                print("Direction: {} - LEFT".format(direction))
            if direction == "RIGHT":
                #Give both motors forward high
                # GPIO.output(In1, GPIO.HIGH)
                # GPIO.output(In2, GPIO.LOW)
                print("Direction: {} - RIGHT".format(direction))
            if direction == "CENTER":
                #Give both motors LOW
                # GPIO.output(In1, GPIO.LOW)
                # GPIO.output(In2, GPIO.LOW)
                print("Direction: {} - CENTER".format(direction))
 
 
while True:
    # asyncio.run(run_client()) # Function used for Python 3.7
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_client())