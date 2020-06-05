# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/thu-coai/ConvLab-2/blob/master/convlab2/util/multiwoz/dbquery.py
"""
import json
import os
import random

__all__ = ['Database']


class Database(object):
    def __init__(self, data_dir):
        super(Database, self).__init__()
        # loading databases
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
        self.dbs = {}
        for domain in domains:
            with open(os.path.join(data_dir, 'db/{}_db.json'.format(domain))) as f:
                self.dbs[domain] = json.load(f)

    def query(self, domain, constraints, ignore_open=True):
        """Returns the list of entities for a given domain
        based on the annotation of the belief state"""
        # query the db
        if domain == 'taxi':
            return [
                {
                    'taxi_colors': random.choice(self.dbs[domain]['taxi_colors']),
                    'taxi_types': random.choice(self.dbs[domain]['taxi_types']),
                    'taxi_phone': ''.join([str(random.randint(1, 9)) for _ in range(11)]),
                }
            ]
        if domain == 'police':
            return self.dbs['police']
        if domain == 'hospital':
            return self.dbs['hospital']

        found = []
        for i, record in enumerate(self.dbs[domain]):
            for key, val in constraints:
                if (
                    val == ""
                    or val == "dont care"
                    or val == 'not mentioned'
                    or val == "don't care"
                    or val == "dontcare"
                    or val == "do n't care"
                ):
                    pass
                else:
                    try:
                        record_keys = [k.lower() for k in record]
                        if key.lower() not in record_keys:
                            continue
                        if key == 'leaveAt':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                            if val1 > val2:
                                break
                        elif key == 'arriveBy':
                            val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                            val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                            if val1 < val2:
                                break
                        # elif ignore_open and key in ['destination', 'departure', 'name']:
                        elif ignore_open and key in ['destination', 'departure']:
                            continue
                        else:
                            if val.strip() != record[key].strip():
                                break
                    except Exception:
                        continue
            else:
                record['Ref'] = '{0:08d}'.format(i)
                found.append(record)

        return found
