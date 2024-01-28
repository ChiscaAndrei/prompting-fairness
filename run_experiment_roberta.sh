#!/bin/bash

EVALUATIONS_REPO=KegBias/evaluations-roberta-experiment1
BASE_MODEL_NAME=roberta-base
DO_TRAINING=true
DO_EVALUATION=true
PROMPT_LENGTH=3

NEUTRAL_PROMPTS=(
    " tree stone lake"
    " animal mountain house"
    " fair water balanced"
    " justice equality human"
    " honest forest gold"
)

GENDERED_PROMPTS=(
    " man water woman"
    " girl fair boy"
    " masculine human feminine"
    " female justice male"
    " husband big queen"
)

FEMALE_BIASED_PROMPTS=(
    " women queen girl"
    " female her feminine"
    " wife feminist mothers"
    " woman miss female"
    " she feminism mothers"
)

if $DO_EVALUATION
then
    python -m src.prompt_tune_sandbox.cli evaluate-model ${BASE_MODEL_NAME} ${EVALUATIONS_REPO} --eval-base-model
    sleep 30s
fi

#for INIT_METHOD in random neutral gendered 
#for INIT_METHOD in gendered femaleBiased
for INIT_METHOD in random neutral gendered femaleBiased
do
    for TRIAL_NO in {1..10}
    do
        EXPERIMENT_NAME=${BASE_MODEL_NAME}_${INIT_METHOD}-init_v${TRIAL_NO}
        REPO_PREFIX=KegBias/

        if [[ $INIT_METHOD != random &&  ( "$TRIAL_NO" -gt 5 ) ]]; then
            continue
        fi

        if [ $INIT_METHOD == random ]; then
            PROMPT_INIT_ARGUMENT=""
        elif [ $INIT_METHOD == neutral ]; then
            PROMPT_INIT_ARGUMENT="--prompt-init-text=${NEUTRAL_PROMPTS[${TRIAL_NO}-1]}"
        elif [ $INIT_METHOD == gendered ]; then
            PROMPT_INIT_ARGUMENT="--prompt-init-text=${GENDERED_PROMPTS[${TRIAL_NO}-1]}"
        elif [ $INIT_METHOD == femaleBiased ]; then
            PROMPT_INIT_ARGUMENT="--prompt-init-text=${FEMALE_BIASED_PROMPTS[${TRIAL_NO}-1]}"
        fi

        if $DO_TRAINING
        then
            if [ $INIT_METHOD == random ]; then
                python -m src.prompt_tune_sandbox.cli train-model\
                    --experiment-name=${EXPERIMENT_NAME}\
                    --prompt-length=${PROMPT_LENGTH}\
                    --loss-type=original_model_probability_options\
                    --gender-specific-options\
                    --model-name=${BASE_MODEL_NAME}\
                    --num-epochs=250\
                    --eval-interval=125\
                    --position-ids-adjustment=none\
                    --no-use-names\
                    --hub-repo-prefix=${REPO_PREFIX}
            else
                python -m src.prompt_tune_sandbox.cli train-model\
                    --experiment-name=${EXPERIMENT_NAME}\
                    --prompt-length=${PROMPT_LENGTH}\
                    --loss-type=original_model_probability_options\
                    --gender-specific-options\
                    --model-name=${BASE_MODEL_NAME}\
                    --num-epochs=250\
                    --eval-interval=125\
                    --position-ids-adjustment=none\
                    "${PROMPT_INIT_ARGUMENT}"\
                    --no-use-names\
                    --hub-repo-prefix=${REPO_PREFIX}
            fi
            sleep 100s
        fi

        if $DO_EVALUATION
        then
            python -m src.prompt_tune_sandbox.cli evaluate-model ${REPO_PREFIX}${EXPERIMENT_NAME} ${EVALUATIONS_REPO}
            sleep 100s
        fi
    done
    sleep 300s
done

