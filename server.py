import asyncio
import websockets
import random
import time
directions = ["LEFT", "RIGHT", "CENTER", "FORWARD", "BACKWARD"]
async def echo(websocket, path):
    async for message in websocket:
        #this will select a random direction and send it
        rand_idx = random.randrange(len(directions))
        rand_dir = directions[rand_idx]
        await websocket.send(rand_dir)
        time.sleep(1)

host = "localhost"
port = 8765
async def run_server():
    server = await websockets.serve(echo, host, port)
    if server:
        print("Server Started Succesfully!")
    await server.wait_closed()
 
while True:
    asyncio.run(run_server())