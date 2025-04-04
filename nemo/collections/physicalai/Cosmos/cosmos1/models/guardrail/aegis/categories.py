# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VIOLENCE = {
    "category": "violence",
    "blocked_reason": """This prompt is blocked due to the presence of violent content.""",
    "instruction": """O1: Violence.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Contain unpleasant vivid violent imagery or violent description that arouses discomfort in the reader.
Can
- Provide information on violence.
- Discuss the topics of violence, and can discuss historical events involving violence.
""",
}

SEXUAL = {
    "category": "sexual",
    "blocked_reason": """This prompt is blocked due to the presence of sexual content.""",
    "instruction": """O2: Sexual.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
- Engage or encourage sexual activity that is considered illegal.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
""",
}

CRIMINAL_PLANNING = {
    "category": "criminal_planning",
    "blocked_reason": """This prompt is blocked due to the presence of criminal planning/confessions content.""",
    "instruction": """O3: Criminal Planning/Confessions.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
- Have a confession of crime that was committed or will be committed in future.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
""",
}

WEAPONS = {
    "category": "weapons",
    "blocked_reason": """This prompt is blocked due to the presence of weapons content.""",
    "instruction": """O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
- Assist or encourage people to execute or own other illegal weapons for example biological agents, or chemical weapons, including instructions on how to create or use them.
Can
- Discuss firearms and the arguments for and against firearm ownership.
""",
}

SUBSTANCE_ABUSE = {
    "category": "substance_abuse",
    "blocked_reason": """This prompt is blocked due to the presence of substance abuse content.""",
    "instruction": """O5: Controlled/Regulated Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
""",
}

SUICIDE = {
    "category": "suicide",
    "blocked_reason": "This prompt is blocked due to the presence of suicide and self harm content.",
    "instruction": """O6: Suicide and Self Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
- Contain confessions or intent of self harm
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
""",
}

CSAM = {
    "category": "csam",
    "blocked_reason": "This prompt contains material related to Child Sexual Abuse",
    "instruction": """O7: Sexual (minor).
Should not
- Assist, engage or encourage intention to engage in sexual activity with an underage/minor.
- Discuss or plan sexual acts, sharing of media of sexual nature, personal information or intention to meet in person for sexual activity with a minor.
- Should not include discussion where private information is shared with underage minors for purposes of acting on them and taking actions that are sexual in nature including meeting or sharing media.
- Should not discuss child abuse or engage, encourage, assist in sexual grooming/trafficking of children.
Can
- Provide information about sex education for use in schools.
""",
}

HATRED = {
    "category": "hatred",
    "blocked_reason": "This prompt is blocked due to the presence of hate speech content.",
    "instruction": """O8: Hate/Identity Hate.
Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their
race, color, ethnicity, religion, national origin, sexual orientation, gender, gender identity, or disability or caste.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
- Should not discuss prejudice towards a group classified as protected according to the law of Germany, UK, USA or any other country.
- Should not contain slurs.
- Should not contain text that is distasteful, offensive, or hateful that causes others to feel grief, anger, or fear.
Can
- Provide information on discrimination.
- Discuss the topics of discrimination, and can discuss historical events involving discrimination.
- Discuss neutral statements referring to group identity and  contextualized hate speech (that talks about hate but does not incite hate).
""",
}

HARASSMENT = {
    "category": "harassment",
    "blocked_reason": "This prompt is blocked due to the presence of harassment content.",
    "instruction": """O10: Harassment.
Should not
- Have content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.
- Engage in harassment, misconduct and unlawful discrimination
- Single out an individual or marginalization and/or retaliation based on protected characteristics like Race, Color, Gender, Sex,
Sexual orientation, Gender identity, and gender expression, National origin, Ethnicity, Disability (including being regarded as disabled),
Religion, Age (40+), Pregnancy (including pregnancy, childbirth or related medical conditions), Genetic information, Military or veteran status, Citizenship status, Political activity, or affiliation,
- Taking or requesting statutorily protected leave Body characteristics.
- Promise to give a benefit, or a threat to retaliate or take an adverse action based on a response to harassment and misconduct.
- Bullying.
- Being derogatory and humiliating toward an individual but not necessarily protected characteristics under law.
- Making rude or insulting comments, demeaning, and objectifying terms toward an individual.
Can
- Discuss negative impacts of harassment and bullying based on protected characteristics like race, color, gender, sex, nationality, within a contextualized setting.
- Provide information on protected characteristics or refers, cites material that describes what protected characteristics are.
- Provide information or refer or cite material that informs what harassment is.
""",
}

THREAT = {
    "category": "threat",
    "blocked_reason": "This prompt is blocked due to the presence of threat content.",
    "instruction": """O11: Threat.
Should not
- Discuss a communication of intent to inflict harm or loss on another person.
- Should not contain discussion of intention of intimidation which  is a tactic used between conflicting parties to make the other timid or psychologically insecure for coercion or control.
- Should not contain intention of knowingly putting another person in fear of bodily injury.
- Should not assist or plan injury, physical or mental damage, act or instance of injury, or material  or any kind of loss to a person.
- Should not contain threats that are forbidden by law as intent to obtain a monetary advantage or to compel a person to act against their will.
Can
- Provide information about threat is and the meaning of threat.
""",
}

PROFANITY = {
    "category": "profanity",
    "blocked_reason": "This prompt is blocked due to the presence of profanity content.",
    "instruction": """O12: Profanity.
Should not
- Contain swear words, curse words, or other obscene or profane language
- This includes offensive words used without any intention to act on them or without any explicit targeted recipient
""",
}

UNSAFE_CATEGORIES = [
    VIOLENCE,
    SEXUAL,
    CRIMINAL_PLANNING,
    WEAPONS,
    SUBSTANCE_ABUSE,
    SUICIDE,
    CSAM,
    HATRED,
    HARASSMENT,
    THREAT,
    PROFANITY,
]
