# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

base_starts = [
    "I", "I'm", "I'll", "I've", "My", "You", "You'll", 
    "And", "But", "So", "He", "He'll", "She", "She'll", 
    "They", "They'll", "We", "We'll", "What", "Where", 
    "When", "Why", "Who", "How", "Can", "Could", 
    "Should", "Would", "May", "Might", "Shall", "Will", 
    "Do", "Does", "Did", "Is", "Are", "Was", 
    "Were", "Have", "Has", "Had", "It", "It's", 
    "Isn't", "Aren't", "Wasn't", "Weren't", "Don't", 
    "Doesn't", "Didn't", "Shouldn't", "Couldn't", "Can't", 
    "Won't", "Wouldn't", "Oh", "Wow", "No", "Yes", "Yeah",
    "Let's", "Now", "Okay", "Alright", "Fine", "Sure", "Not", "Maybe",
    "Soon", "Thanks", "Thank", "Hey", "Hi", 
    "Hello", "Well", "Indeed", "Frankly", "Clearly", "Obviously",
    "Unfortunately", "Fortunately", "Interestingly", "Actually", 
    "Basically", "Ultimately", "Simply", "Sure", "'Cause", 
    "Perhaps", "Of", "For", "For,", 
    "However", "Although", "Since", "Because", "Exactly", 
    "Meanwhile", "Later", "First", "Second", 
    "Third", "Finally", "Lastly", "Then", 
    "Afterward", "Once", "Before", "Until", 
    "Even", "Still", "Just", "Sometimes", 
    "Often", "Usually", "Rarely", "Never", 
    "Always", "Every", "Each", "Some", 
    "Many", "Most", "Few", "Several", 
    "All", "Both", "Either", "Neither", 
    "One", "Another", "This", "That", 
    "These", "Those", "There", "Here", 
    "Somehow", "Someone", "Something", "Somewhere", 
    "Nothing", "Nobody", "None", "Everything", 
    "Everyone", "Everywhere", "Anybody", "Anything", 
    "Anyone"
]

# Generate a list with both original and comma versions
COMMON_SENTENCE_STARTS = base_starts + [word + "," for word in base_starts]
