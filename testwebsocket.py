import asyncio
import websockets

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws/debate"
    async with websockets.connect(uri) as websocket:
        await websocket.send("hello")
        response = await websocket.recv()
        print("Received:", response)

asyncio.run(test_ws())
