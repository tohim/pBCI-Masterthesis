import socket
import threading
import time

# MATLAB Connection
MATLAB_HOST = "127.0.0.1"
MATLAB_PORT = 65432

running = False
current_speed = "LOW"  # Default

def listen_for_matlab(conn):
    global running, current_speed
    while True:
        data = conn.recv(1024).decode('utf-8').strip()
        if data == "STOP":
            print("[MATLAB] STOP received – pausing robot (simulated).")
            running = False
            conn.sendall(b"ACK: STOP")
        elif data == "CONTINUE":
            print("[MATLAB] CONTINUE received – resuming robot (simulated).")
            running = True
            conn.sendall(b"ACK: CONTINUE")
        elif data in ["LOW", "HIGH"]:
            current_speed = data
            print(f"[MATLAB] Speed profile set to: {current_speed} (simulated).")
            conn.sendall(b"ACK: SPEED")
        else:
            print(f"[MATLAB] Unknown command: {data}")
            conn.sendall(b"ACK: UNKNOWN")

def main():
    global running, current_speed

    # Start MATLAB TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.bind((MATLAB_HOST, MATLAB_PORT))
        server_sock.listen(1)
        print("Waiting for MATLAB to connect (test mode)…")
        conn, addr = server_sock.accept()
        print("MATLAB connected:", addr)

        # Start a thread to listen for MATLAB commands
        threading.Thread(target=listen_for_matlab, args=(conn,), daemon=True).start()

        # Main loop to simulate robot actions
        while True:
            if running:
                print(f"[SIMULATION] Robot would move with {current_speed} speed profile…")
            else:
                print("[SIMULATION] Robot is paused – waiting for CONTINUE from MATLAB.")
            time.sleep(2)

if __name__ == "__main__":
    main()
