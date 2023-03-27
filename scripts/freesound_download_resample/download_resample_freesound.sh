# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#!/bin/bash

# This is bash script actually run the downloading and resampling script.
# See instructions in freesound_download.py

# Change this arguments if you want
page_size=100   # Number of sounds per page
max_samples=200 # Maximum number of sound samples
min_filesize=0  # Minimum filesize allowed (in MB)
max_filesize=100 # Maximum filesize allowed (in MB)

if [[ $# -ne 3 ]]; then
  echo "Require number of all files | data directory | resample data directory as arguments to the script"
  exit 2
fi


NUM_ALL_FILES=$1
DATADIR=$2
RESAMPLE_DATADIR=$3


if [ ! -d "$DATADIR" ]; then
    echo "Creating dir $DATADIR"
    mkdir -p "$DATADIR"
fi

if [ ! -d "$RESAMPLE_DATADIR" ]; then
    echo "Creating dir $RESAMPLE_DATADIR"
    mkdir -p "$RESAMPLE_DATADIR"
fi

# we just need background categories for constructing dataset, feel free to include other (speech) categories for testing and training your VAD model
# background
categories=(
    "Air brake"
    "Static"
    "Acoustic environment"
    "Distortion"
    "Tape hiss"
    "Hubbub"
    "Vibration"
    "Cacophony"
    "Throbbing"
    "Reverberation"
    "Inside, public space"
    "Inside, small room"
    "Echo"
    "Outside, rural"
    "Outside, natural"
    "Outside, urban"
    "Outside, manmade"
    "Car"
    "Bus"
    "Traffic noise"
    "Roadway noise"
    "Truck"
    "Emergency vehicle"
    "Motorcycle"
    "Aircraft engine"
    "Aircraft"
    "Helicopter"
    "Bicycle"
    "Skateboard"
    "Subway, metro, underground"
    "Railroad car"
    "Train wagon"
    "Train"
    "Sailboat"
    "Rowboat"
    "Ship"
)


WAV_FILECOUNT="$(find $DATADIR  -name '*.wav' -type f | wc -l)"
FLAC_FILECOUNT="$(find $DATADIR  -name '*.flac' -type f | wc -l)"
FILECOUNT="$((WAV_FILECOUNT + FLAC_FILECOUNT))"
echo "File count: " $FILECOUNT


while((FILECOUNT <= NUM_ALL_FILES))
do
  for category in "${categories[@]}"
  do
    python freesound_download.py --data_dir "${DATADIR}" --category "${category}" --page_size "${page_size}" --max_samples "${max_samples}" --min_filesize "${min_filesize}" --max_filesize "${max_filesize}"
    ret=$?
    if [ $ret -ne 0 ]; then
        exit 1
    fi
  done
  
  WAV_FILECOUNT="$(find $DATADIR  -name '*.wav' -type f | wc -l)"
  FLAC_FILECOUNT="$(find $DATADIR  -name '*.flac' -type f | wc -l)"
  FILECOUNT="$((WAV_FILECOUNT + FLAC_FILECOUNT))"
  echo "Current file count is: " $FILECOUNT
done

# RESAMPLE
echo "Got enough data. Start resample!"
python freesound_resample.py --data_dir="${DATADIR}" --resampled_dir="${RESAMPLE_DATADIR}"

echo "Done resample data!"
