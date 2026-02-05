import csv
import os
from datetime import datetime

class DataLogger:
    """
    Gestisce il logging dei dati su file CSV.
    Colonne: timestamp, HR, arousal, method, rmssd
    """
    def __init__(self):
        self.file = None
        self.writer = None
        self.active = False

    def start(self, directory: str):
        """
        Inizia una nuova sessione di logging in 'directory', crea
        il file session_{timestamp}.csv e scrive l'header.
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory non valida: {directory}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'session_{timestamp}.csv'
        filepath = os.path.join(directory, filename)
        f = open(filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(f)
        # header include anche rmssd
        self.writer.writerow(['timestamp', 'HR', 'arousal', 'method', 'rmssd'])
        self.file = f
        self.active = True

    def log(self, hr: int, arousal: float, method: str, rmssd: float = None):
        """
        Aggiunge una riga con timestamp ISO, HR, valore arousal, metodo ('actual' o 'estimated')
        e rMSSD (solo per 'actual', 0 per 'estimated').
        """
        print("RMSSD: "+str(rmssd))
        if not self.active or self.writer is None:
            return
        ts = datetime.now().isoformat()
        # rMSSD per dati reali, 0 per stimati
        if method == 'actual' and rmssd is not None:
            rmssd_str = f'{rmssd:.4f}'
        else:
            rmssd_str = '0'
        self.writer.writerow([ts, hr, f'{arousal:.2f}', method, rmssd_str])

    def stop(self):
        """
        Termina la sessione di logging e chiude il file.
        """
        if self.active and self.file:
            try:
                self.file.close()
            finally:
                self.active = False
                self.file = None
                self.writer = None
