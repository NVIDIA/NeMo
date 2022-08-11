import os
import re
import socket


def pause_and_prime_dns_connections() -> None:
    if int(os.environ.get("GROUP_RANK")) > 0:
        time.sleep(20)
        prime_dns_connections()
    elif int(os.environ.get("LOCAL_RANK")) != 0:
        time.sleep(10)

def prime_dns_connections() -> None:
    me = "worker" + os.environ.get("GROUP_RANK") + ":" + os.environ.get("RANK")
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = int(os.environ.get("MASTER_PORT"))
    print(f"SPDNS: {me} Connecting to {master_addr}:{master_port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (master_addr, master_port)
    timeout = time.time() + 300
    connected = False
    while not connected:
        try:
            sock.connect(server_address)
            connected = True
        except Exception:
            time.sleep(2)
        if time.time() > timeout:
            print(f"{me} couldnt connect to {master_addr}:{master_port} timed out! (300s)")
            sys.exit(110)
    print(f"SPDNS: {me} connected to {master_addr}:{master_port}")
    sock.close()

if __name__ == "__main__":
    pause_and_prime_dns_connections()