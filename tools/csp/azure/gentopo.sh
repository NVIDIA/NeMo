#! /bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

declare -A gpus
declare -A hcas

for i in /sys/bus/pci/drivers/nvidia/*/numa_node; do
gpus[$(< $i)]+=" $(awk -F/ '{print $7}' <<< $i) "
done
for i in /sys/bus/pci/drivers/mlx5_core/*/numa_node; do
hcas[$(< $i)]+=" $(awk -F/ '{print $7}' <<< $i) "
done

gpus_node0=(${gpus[0]})
gpus_node1=(${gpus[1]})
gpus_node2=(${gpus[2]})
gpus_node3=(${gpus[3]})
hcas_node0=(${hcas[0]})
hcas_node1=(${hcas[1]})
hcas_node2=(${hcas[2]})
hcas_node3=(${hcas[3]})

cat <<EOF
<system version="1">
  <cpu numaid="0" affinity="0000ffff,0000ffff" arch="x86_64" vendor="AuthenticAMD" familyid="143" modelid="49">
    <pci busid="ffff:ff:01.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="${gpus_node0[0]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node0[0]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${gpus_node0[1]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node0[1]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
    <pci busid="ffff:ff:02.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="${gpus_node1[0]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node1[0]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${gpus_node1[1]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node1[1]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
    <pci busid="ffff:ff:03.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="${gpus_node2[0]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node2[0]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${gpus_node2[1]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node2[1]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
    <pci busid="ffff:ff:04.0" class="0x060400" link_speed="16 GT/s" link_width="16">
      <pci busid="${gpus_node3[0]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node3[0]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${gpus_node3[1]}" class="0x030200" link_speed="16 GT/s" link_width="16"/>
      <pci busid="${hcas_node3[1]}" class="0x020700" link_speed="16 GT/s" link_width="16"/>
    </pci>
  </cpu>
</system>
EOF
