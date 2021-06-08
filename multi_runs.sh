#-----Train 3 Normal teachers
for seeds in 10 20
do
    python main.py --experiment=0.1 --seed=$seeds --gpu="0,1" --log_offline
done

#-----Train 3 Augmentation teachers
for seeds in 20 30
do
    python main.py --experiment=0.2 --seed=$seeds --gpu="0,1" --log_offline
done

#-----Train CBD_ensemble_K
for seeds in 1
do
    for alphas in 0.2 0.4 0.8
    do
        betas in 50 100 200
        do
            python main.py --experiment=0.3 --alpha=$alphas --beta=$betas --seed=$seeds --gpu="0,1" --log_offline --normal_teacher="10,20" --aug_teacher="20,30"
        done
    done
done
