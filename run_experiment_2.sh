#!/bin/bash

EVALUATIONS_REPO=KegBias/evaluations-experiment2
BASE_MODEL_NAME=bert-base-uncased
DO_TRAINING=true
DO_EVALUATION=true
NO_GENDER_SPECIFIC=true
NAMES_IN_TEMPLATES=true
PROMPT_LENGTH=3

FEMALE_BIASED_PROMPTS=(
    "women queen girl"
    "female priestess feminine"
    "wife feminist mothers"
    "woman miss suffrage"
    "she feminism mothers"
)


if $DO_EVALUATION
then
    python -m src.prompt_tuning_debias.cli evaluate-model ${BASE_MODEL_NAME} ${EVALUATIONS_REPO} --eval-base-model
fi

for TRIAL_NO in {1..5}
do
    REPO_PREFIX=KegBias/
    PROMPT_INIT_ARGUMENT="--prompt-init-text=${FEMALE_BIASED_PROMPTS[${TRIAL_NO}-1]}"

    if $NO_GENDER_SPECIFIC
    then
        EXPERIMENT_NAME=${BASE_MODEL_NAME}_random-init_no-gender-specific_v${TRIAL_NO}

        if $DO_TRAINING
        then
            python -m src.prompt_tuning_debias.cli train-model\
                --experiment-name=${EXPERIMENT_NAME}\
                --prompt-length=${PROMPT_LENGTH}\
                --loss-type=original_model_probability_options\
                --no-gender-specific-options\
                --model-name=${BASE_MODEL_NAME}\
                --num-epochs=250\
                --eval-interval=125\
                --position-ids-adjustment=none\
                "${PROMPT_INIT_ARGUMENT}"\
                --no-use-names\
                --hub-repo-prefix=${REPO_PREFIX}
        fi

        if $DO_EVALUATION
        then
            python -m src.prompt_tuning_debias.cli evaluate-model ${REPO_PREFIX}${EXPERIMENT_NAME} ${EVALUATIONS_REPO}
        fi
    fi

    if $NAMES_IN_TEMPLATES
    then
        EXPERIMENT_NAME=${BASE_MODEL_NAME}_random-init_with-names_v${TRIAL_NO}

        if $DO_TRAINING
        then
            python -m src.prompt_tuning_debias.cli train-model\
                --experiment-name=${EXPERIMENT_NAME}\
                --prompt-length=${PROMPT_LENGTH}\
                --loss-type=original_model_probability_options\
                --gender-specific-options\
                --model-name=${BASE_MODEL_NAME}\
                --num-epochs=40\
                --eval-interval=20\
                --position-ids-adjustment=none\
                "${PROMPT_INIT_ARGUMENT}"\
                --use-names\
                --hub-repo-prefix=${REPO_PREFIX}
        fi

        if $DO_EVALUATION
        then
            python -m src.prompt_tuning_debias.cli evaluate-model ${REPO_PREFIX}${EXPERIMENT_NAME} ${EVALUATIONS_REPO}
        fi
    fi
done

