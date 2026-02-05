import asyncio
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name and 'HW9' in d.name:
            print(f"👉 Trovato: {d.name} [{d.address}]")

asyncio.run(scan())