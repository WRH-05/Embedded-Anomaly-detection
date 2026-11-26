import serial
import serial.tools.list_ports
import time
import os

BAUD_RATE = 115200
SAMPLES_PER_CLASS = 100
OUTPUT_FILE = "dataset_P09_FINAL.csv"

CLASSES = {
    "1": "CALME",
    "2": "VIBRATION_NORMALE",
    "3": "VIBRATION_ANORMALE"
}

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[!] No COM ports detected! Check USB connection.")
        exit(1)
    print("\n--- Available COM Ports ---")
    for idx, p in enumerate(ports):
        print(f"[{idx}] {p.device} - {p.description}")
    choice = input("\nSelect port index (default 0): ").strip()
    return ports[int(choice) if choice else 0].device

def countdown(seconds):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i}... (Get in position!)", end="\r")
        time.sleep(1)
    print("GO! Recording now...                 ")

def main():
    port = select_serial_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)  # Wait for ESP32 reset

    # Create CSV with strict PDF-compliant header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("timestamp,RMS_x,RMS_y,RMS_z,peak_acc,frequence_dominante,energie,label\n")
        print(f"[*] Initialized file: {OUTPUT_FILE}")
    else:
        print(f"[*] Appending to existing file: {OUTPUT_FILE}")

    while True:
        print("\n" + "="*40)
        print(" SELECT CLASS TO RECORD (100 samples each):")
        print(" [1] CALME (Sensor sitting untouched on table)")
        print(" [2] VIBRATION_NORMALE (Continuous finger tapping on desk)")
        print(" [3] VIBRATION_ANORMALE (Hard knocks / table pounding)")
        print(" [Q] Quit and finalize")
        print("="*40)

        choice = input("Enter choice (1, 2, 3, or Q): ").strip().upper()
        if choice == 'Q':
            break
        if choice not in CLASSES:
            print("[!] Invalid choice. Try again.")
            continue

        label = CLASSES[choice]
        print(f"\n[>>>] PREPARING FOR: {label}")
        countdown(3)  # 3-second buffer to remove hand-transient spikes

        ser.reset_input_buffer()
        collected = 0

        with open(OUTPUT_FILE, "a") as f:
            while collected < SAMPLES_PER_CLASS:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Verify exactly 6 commas (7 fields: timestamp + 6 features)
                if line and line.count(',') == 6:
                    f.write(f"{line},{label}\n")
                    f.flush()
                    collected += 1
                    percent = (collected / SAMPLES_PER_CLASS) * 100
                    print(f"[{label}] {collected:03d}/{SAMPLES_PER_CLASS} ({percent:5.1f}%) | {line}")

        print(f"\n[DONE] Successfully captured 100 samples for {label}!")

    ser.close()
    print(f"\n[SUCCESS] Dataset saved to {OUTPUT_FILE} (Total: 300 clean samples).")

if __name__ == "__main__":
    main()