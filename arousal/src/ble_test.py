import asyncio
from bleak import BleakScanner, BleakClient

CHAR_UUID = '00002a37-0000-1000-8000-00805f9b34fb'

def callback(sender, data):
    print(f"Ricevuto: {list(data)}")

async def main():
    devices = await BleakScanner.discover()
    target = next((d for d in devices if d.name and ('Coospo' in d.name or 'HW9' in d.name)), None)
    if not target:
        print("HW9 non trovato.")
        return

    async with BleakClient(target.address) as client:
        print("Connesso!")
        await client.start_notify(CHAR_UUID, callback)
        await asyncio.sleep(30)  # ricevi dati per 30 secondi

asyncio.run(main())
