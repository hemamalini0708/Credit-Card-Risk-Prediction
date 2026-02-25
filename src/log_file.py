import logging
import os


def phase_1(acc):
    log = logging.getLogger(acc)
    log.setLevel(logging.INFO)

    # Create logs folder in project directory
    os.makedirs("logs", exist_ok=True)

    # Avoid duplicate handlers
    if not log.handlers:
        file_path = os.path.join("logs", f"{acc}.log")
        handler = logging.FileHandler(file_path, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        log.addHandler(handler)

    return log