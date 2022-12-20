# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is used to `pause_and_prime_dns_connections` in BCP platform.
"""

import os
import re
import socket
import time


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
