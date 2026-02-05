import sys
import asyncio
from PySide6.QtWidgets import QApplication
#from ui_main import MainWindow
from ui_main_alt import MainWindow
from ble_monitor import BleMonitor
from qasync import QEventLoop

async def start_app():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    monitor = BleMonitor(window)
    window.set_monitor(monitor)

    # rimuovo il connect automatico: ora parte quando premi "Start Visualization"
    # await window.try_connect()
    asyncio.create_task(monitor.run())

    with loop:
        loop.run_forever()

if __name__ == '__main__':
    asyncio.run(start_app())
