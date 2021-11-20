import os
import shutil

SOURCE_DIR='results/generate_IDEAS_samples_forFID'

DATASET_NAMEs=['Bedroom','Church', 'FFHQ']
Ns=[1,2]
SIGMAs=[1,2,3]
DELTAs=['0.0', '0.25', '0.5']

TARGET_DIR='results/IDEAS_samples_for_Steganalysis'
for sigma in SIGMAs:
    for dataset in DATASET_NAMEs:
        for N in Ns:
            for delta in DELTAs:
                target_path=f'{TARGET_DIR}/for_ROC_sigma={sigma}/IDEAS_N={N}_delta={delta}/{dataset}_sample_500'
                os.makedirs(target_path,exist_ok=True)
                source_path=f'{SOURCE_DIR}/IDEAS_{dataset}_N={N}_lambdaREC=10/sigma={sigma}_delta={delta}'
                for i in range(1,501):
                    source_img_path=f'{source_path}/{i:06d}.png'
                    target_img_path=f'{target_path}/{i:06d}.png'
                    shutil.copyfile(source_img_path,target_img_path)
                print(f'Done for_ROC_sigma={sigma}/IDEAS_N={N}_delta={delta}/{dataset}_sample_500')



