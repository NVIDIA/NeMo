if __name__ == "__main__":
    vars = {
        "RUN_MODEL": "gpt3",
        "RUN_MODEL_SIZE": "126m",
        "RUN_NAME_SUFFIX": "SP",
        "TP_SIZE": "2",
        "PP_SIZE": "2",
        "PP_SPLIT_RANK": "",
        "NUM_NODES": "2",
        "MAX_STEPS": "100",
        "TIME_LIMIT": "00:40:00",
        "TEST_LEVEL": "L3",
        "ADDITIONAL_PARAMS": "training.model.sequence_parallel=True"
    }
    if 'RUN_NAME_SUFFIX' in vars and vars['RUN_NAME_SUFFIX']:
        suffix = "_" + vars['RUN_NAME_SUFFIX']
    else:
        suffix = ""

    mp_string = \
    f"""TP_SIZE: {vars['TP_SIZE']}
                PP_SIZE: {vars['PP_SIZE']}"""
    if vars["PP_SPLIT_RANK"]:
        mp_string += f"\n                PP_SPLIT_RANK: {vars['PP_SPLIT_RANK']}"

    for stage in ["train", "eval", "convert", "finetune", "prompt_learn"]:
        vars['RUN_STAGE'] = stage
        vars[
            'RUN_NAME'] = f"{vars['RUN_STAGE']}_{vars['RUN_MODEL']}_{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}"

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if vars['RUN_STAGE'] == "convert":
            vars['CONVERT_NAME'] = vars['RUN_NAME'] + suffix

            text = \
                f"""
            convert.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: {vars['RUN_STAGE']}
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                UPSTREAM_RUN_NAME: {vars['TRAIN_NAME']}
                {mp_string}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - train.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps{suffix}
            """
            text = text.replace("\n            ", "\n")
            print(text)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        elif vars['RUN_STAGE'] == "train":
            vars['RUN_NAME'] = f"{vars['RUN_NAME']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps"
            vars['TRAIN_NAME'] = vars['RUN_NAME'] + suffix

            text = \
                f"""
            train.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: {vars['RUN_STAGE']}
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                {mp_string}
                NUM_NODES: {vars['NUM_NODES']}
                MAX_STEPS: {vars['MAX_STEPS']}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
                ADDITIONAL_PARAMS: "{vars['ADDITIONAL_PARAMS']}"
              needs:
                - build-BigNLP
            """
            text = text.replace("\n            ", "\n")
            print(text)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        elif vars['RUN_STAGE'] == "prompt_learn":
            vars['TEST_TASK'] = "squad"
            vars['RUN_NAME'] = f"{vars['RUN_NAME']}_{vars['NUM_NODES']}node_{vars['TEST_TASK']}"
            vars['PROMPT_NAME'] = vars['RUN_NAME'] + suffix

            text = \
                f"""
            prompt_learn.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps_{vars['TEST_TASK']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: {vars['RUN_STAGE']}
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                TEST_TASK: {vars['TEST_TASK']}
                {mp_string}
                NUM_NODES: {vars['NUM_NODES']}
                MAX_STEPS: {vars['MAX_STEPS']}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - convert.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}{suffix}
            """
            text = text.replace("\n            ", "\n")
            print(text)

            text = \
                f"""
            eval.{vars['RUN_MODEL']}.prompt_{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['TEST_TASK']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: eval
                RUN_MODEL: prompt_{vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                TEST_TASK: {vars['TEST_TASK']}
                {mp_string}
                NUM_NODES: {vars['NUM_NODES']}
                PROMPT_LEARN_MODEL_DIR: {vars['PROMPT_NAME']}
                CONVERT_MODEL_DIR: {vars['CONVERT_NAME']}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - prompt_learn.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps_{vars['TEST_TASK']}{suffix}
            """

            text = text.replace("\n            ", "\n")
            print(text)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        elif vars['RUN_STAGE'] == "finetune" and vars['RUN_MODEL'] in ["t5", "mt5"]:
            if vars['RUN_MODEL'] == "t5":
                vars['TEST_TASK'] = "squad"
            elif vars['RUN_MODEL'] == "mt5":
                vars['TEST_TASK'] = "xquad"
            vars[
                'RUN_NAME'] = f"{vars['RUN_NAME']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps_{vars['TEST_TASK']}"
            vars['FINETUNE_NAME'] = vars['RUN_NAME'] + suffix

            text = \
                f"""
            finetune.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps_{vars['TEST_TASK']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: {vars['RUN_STAGE']}
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                TEST_TASK: {vars['TEST_TASK']}
                {mp_string}
                NUM_NODES: {vars['NUM_NODES']}
                MAX_STEPS: {vars['MAX_STEPS']}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - convert.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}{suffix}
            """
            text = text.replace("\n            ", "\n")
            print(text)

            text = \
                f"""
            eval.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['TEST_TASK']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: eval
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                TEST_TASK: {vars['TEST_TASK']}
                {mp_string}
                NUM_NODES: {vars['NUM_NODES']}
                FINETUNE_JOB_DIR: {vars['FINETUNE_NAME']}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - finetune.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps_{vars['TEST_TASK']}{suffix}
            """
            text = text.replace("\n            ", "\n")
            print(text)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        elif vars['RUN_STAGE'] == "eval" and vars['RUN_MODEL'] == "gpt3":
            if vars['RUN_MODEL'] == "t5":
                vars['TEST_TASK'] = "squad"
            elif vars['RUN_MODEL'] == "mt5":
                vars['TEST_TASK'] = "xquad"
            elif vars['RUN_MODEL'] == "gpt3":
                vars['TEST_TASK'] = "lambada"
            else:
                vars['TEST_TASK'] = "squad"
            vars['RUN_NAME'] = f"{vars['RUN_NAME']}_{vars['TEST_TASK']}"
            vars['EVAL_NAME'] = vars['RUN_NAME'] + suffix

            text = \
                f"""
            eval.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['TEST_TASK']}{suffix}:
              <<: *bignlp-LUNA-test-LAUNCHER-refactored
              variables:
                <<: [*VARS, *LUNA_VARS]
                RUN_STAGE: {vars['RUN_STAGE']}
                RUN_MODEL: {vars['RUN_MODEL']}
                RUN_MODEL_SIZE: {vars['RUN_MODEL_SIZE']}
                RUN_NAME_SUFFIX: "{vars['RUN_NAME_SUFFIX']}"
                TEST_TASK: {vars['TEST_TASK']}
                TRAIN_JOB_NAME: {vars['TRAIN_NAME']}
                {mp_string}
                TIME_LIMIT: "{vars['TIME_LIMIT']}"
                TEST_LEVEL: {vars['TEST_LEVEL']}
              needs:
                - train.{vars['RUN_MODEL']}.{vars['RUN_MODEL_SIZE']}_tp{vars['TP_SIZE']}_pp{vars['PP_SIZE']}_{vars['NUM_NODES']}node_{vars['MAX_STEPS']}steps{suffix}
            """
            text = text.replace("\n            ", "\n")
            print(text)

