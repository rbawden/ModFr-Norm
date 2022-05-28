#!/bin/bash
#SBATCH -C v100-32g
#SBATCH -A ncm@v100
#SBATCH --gres=gpu:1
##SBATCH --mem=32G 
#SBATCH --cpus-per-task=10           # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)  
#SBATCH --job-name=safe   # nom du job
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=ft_%j.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=ft_%j.out   # Nom du fichier d'erreur (ici commun avec la sortie)
##SBATCH --array=0-30%1         # 6 travaux ayant les indices 0, 2, 4, 6, 8, et 10

#echo "### Running $SLURM_JOB_NAME ###"

#set -x
cd ${SLURM_SUBMIT_DIR}

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    thisscript=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
    thisdir=`dirname $thisscript`
else
    # otherwise: started with bash. Get the real location.
    thisdir=`realpath $(dirname $0)`
fi
maindir=/gpfswork/rech/ncm/ulv12mq/ModFr-Normalisation

#num=${SLURM_ARRAY_TASK_ID}
#if [ $num -gt 0 ]; then
#resume_param="--resume_from_checkpoint True"
#fi
#if [ "$num" -lt "10" ]; then
#    num="0$num"
#fi

train_file=$maindir/data/raw/train/train.finalised.src-trg.json
valid_file=$maindir/data/raw/dev/dev.finalised.src-trg.json

modeldir=$maindir/mt-models/hf-transformer-ft-2

tok_name=rbawden/modern_french_normalisation  #$maindir/hf-conversion/modern_french_normalisation
model_name=$tok_name

CUDA_LAUNCH_BLOCKING=1 WANDB_DISABLED=true TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
	      python $maindir/hf-finetuning/run_translation.py \
	      --tokenizer_name $tok_name \
	      --model_name_or_path $model_name \
	      --source_lang src --target_lang trg \
	      --file_source_lang src --file_target_lang trg \
	      --preprocessing_num_workers 10 \
	      --train_file $train_file --validation_file $valid_file \
	      --output_dir $modeldir/model/ --do_train \
	      --per_device_train_batch_size 4 \
	      --gradient_accumulation_steps 1 \
	      --num_train_epochs 500 \
	      --ignore_pad_token_for_loss \
	      --logging_dir $modeldir/log --logging_strategy steps --logging_steps 50 --log_level debug \
	      --save_strategy steps --save_steps 10000 \
	      --seed 1 --remove_unused_columns $resume_param --sortish_sampler \
	      --adafactor True --learning_rate 0.0001 --lr_scheduler_type constant
#	      --max_source_length 384 --max_target_length 384 \
