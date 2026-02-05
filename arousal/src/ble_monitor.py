import asyncio
from bleak import BleakScanner, BleakClient

SERVICE_UUID = '0000180d-0000-1000-8000-00805f9b34fb'
CHAR_UUID    = '00002a37-0000-1000-8000-00805f9b34fb'

class BleMonitor:
    def __init__(self, ui_window):
        self.ui = ui_window
        self.client = None

    async def connect_and_notify(self):
        # Pulisci una vecchia connessione, se presente
        await self.disconnect_and_cleanup()

        devices = await BleakScanner.discover()
        target = next((d for d in devices if 'HW9' in (d.name or '')), None)
        if not target:
            raise Exception('Fascia HW9 non trovata')

        # Crea un nuovo client e connetti
        self.client = BleakClient(target.address)
        await self.client.connect()
        await self.client.start_notify(CHAR_UUID, self._hr_callback)

    async def disconnect_and_cleanup(self):
        """Ferma notify e disconnette, se necessario."""
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(CHAR_UUID)
            except Exception:
                pass
            try:
                await self.client.disconnect()
            except Exception:
                pass
        self.client = None


    def _hr_callback(self, sender, data: bytearray):
        try:
            hr = data[1]
            rr_list = [
                int.from_bytes(data[i:i+2], 'little') / 1024
                for i in range(2, len(data), 2)
            ]
            # thread-safe via segnale Qt
            self.ui.data_signal.emit(hr, rr_list)
        except Exception as e:
            print(f"[ERROR] Parsing HRM: {e}")

    async def run(self):
        """Mantieni vivo l’event loop BLE se necessario."""
        while True:
            await asyncio.sleep(1)
