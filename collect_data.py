import serial
import serial.tools.list_ports
import time
import os

# --- Configurations ---
BAUD_RATE = 115200
SAMPLES_PER_CLASS = 100  # The document requires >= 80 (100 gives an ideal buffer)
OUTPUT_FILE = "dataset_P09.csv"

CLASSES = {
    "1": "CALME",
    "2": "VIBRATION_NORMALE",
    "3": "VIBRATION_ANORMALE"
}

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[!] No serial ports detected! Connect your ESP32-S3.")
        exit(1)
    
    print("\n--- Available COM Ports ---")
    for idx, p in enumerate(ports):
        print(f"[{idx}] {p.device} - {p.description}")
    
    choice = input("\nSelect port index (default 0): ").strip()
    idx = int(choice) if choice else 0
    return ports[idx].device

def main():
    port = select_serial_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2) # Allow ESP32 reboot on connection
    
    # Initialize CSV header if file doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w") as f:
            f.write("timestamp,RMS_x,RMS_y,RMS_z,peak_acc,frequence_dominante,energie,label\n")
        print(f"[*] Created new file: {OUTPUT_FILE}")
    else:
        print(f"[*] Appending to existing file: {OUTPUT_FILE}")

    while True:
        print("\n===============================")
        print("  SELECT CLASS TO RECORD:")
        print("  [1] CALME (Board still on table)")
        print("  [2] VIBRATION_NORMALE (Gentle continuous table tapping)")
        print("  [3] VIBRATION_ANORMALE (Hard shocks / knocking)")
        print("  [Q] Quit and finalize dataset")
        print("===============================")
        
        choice = input("Enter choice (1, 2, 3, or Q): ").strip().upper()
        if choice == 'Q':
            break
        if choice not in CLASSES:
            print("[!] Invalid option. Try again.")
            continue
            
        current_label = CLASSES[choice]
        print(f"\n[>>>] RECORDING FOR CLASS: {current_label}")
        print(f"Get ready... collecting {SAMPLES_PER_CLASS} windows (1 second each).")
        
        collected = 0
        ser.reset_input_buffer()

        with open(OUTPUT_FILE, "a") as f:
            while collected < SAMPLES_PER_CLASS:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # Verify valid data frame (expecting 6 commas = 7 values)
                if line and line.count(',') == 5:
                    f.write(f"{line},{current_label}\n")
                    f.flush()
                    collected += 1
                    percent = (collected / SAMPLES_PER_CLASS) * 100
                    print(f"[{current_label}] Progress: {collected}/{SAMPLES_PER_CLASS} ({percent:.1f}%) | Vector: {line}")

        print(f"\n[OK] Completed class '{current_label}' with {SAMPLES_PER_CLASS} samples!")

    ser.close()
    print(f"\n[SUCCESS] Dataset completely saved in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()