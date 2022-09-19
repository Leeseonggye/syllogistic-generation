import argparse
from datetime import datetime

def make_parser(is_jupyter = False) :
    parser = argparse.ArgumentParser(description='BART Train for Syllogistic Generation')
    
    ### Wandb Setting
    parser.add_argument(
        '-project_name',
        type = str,
        help = 'Project name',
        default = 'BART Fine-Tuning Generation Only')
    
    parser.add_argument(
        '-group_name',
        type = str,
        help = 'Group name for K-Fold Validation',
        default = 'Debugging')

    ### Path & Seed Setting
    parser.add_argument(
        '-seed',
        type=int,
        default=42)

    parser.add_argument(
        '-data_path', 
        type = str, 
        default = "/datasets/",
        help='학습 데이터 위치')

    parser.add_argument(
        '-model_path', 
        type = str,
        default = "/code/syllogistic-generation/gpt2/finetune/model_log",
        help = '모델 저장할 위치')

    parser.add_argument(
        '-generation_path',
        type = str,
        default = "/code/syllogistic-generation/gpt2/finetune/generation_log",
        help = '생성된 텍스트 저장할 위치')

    parser.add_argument(
        '-datetime',
        type = str,
        default = datetime.now().strftime("%Y%m%d_%H%M%S"),
        help = 'generation log를 위한 시간 표시')
    
    parser.add_argument(
        '-load_model_path',
        type = str,
        default = 'facebook/bart-base')

    parser.add_argument(
        '-load_tokenizer_path',
        type = str,
        default = 'facebook/bart-base')
    
    # Train Hyperparameter

    parser.add_argument(
        '-gpu_id',
        type = int,
        default = 0)

    parser.add_argument(
        '-num_epochs',
        type = int,
        default = 10)

    parser.add_argument(
        '-num_iteration',
        type = int,
        default = -1,
        help = "학습을 종료할 목표 이터레이션, epoch보다 우선시 됨")

    parser.add_argument(
        '-max_len',
        type = int,
        default = 128,
        help = "모델의 max len을 따르되 필요 시 수정")
    
    parser.add_argument(
        '-batch_size',
        type = int,
        default = 16)
    
    parser.add_argument(
        '--accumulation_steps',
        type = int,
        default = 1,
        help = "gradient accumulation step")

    parser.add_argument(
        "-learning_rate",
        type = float,
        default = 3e-5)

    parser.add_argument(
        '-valid_batch_size',
        type = int,
        default = 16,
        help = "validation batch size")

    parser.add_argument(
        '-train_mode',
        type = str,
        default = "kfold",
        help = "K-Fold validation or FineTuning",
        choices = ['kfold', 'finetuning'])

    parser.add_argument(
        '-augment_mode',
        type = lambda x : x.lower() == 'true',
        default = "False",
        help = "useage of augmented data")

    ### Logging
    parser.add_argument(
        '-fold_name',
        type = str,
        default = None,
        help = "로깅 시 사용되는 이름 ")

    # parser.add_argument(
    #     '-save_epochs',
    #     type = lambda x : x.lower() == 'true',
    #     default = "False",
    #     help = "매 에폭마다 모델, 생성된 텍스트를 저장할지 여부")
    
    parser.add_argument(
        '-valid_epochs',
        type = lambda x : x.lower() == 'true',
        default = "True",
        help = "매 에폭마다 평가를 수행할지 여부")
    
    parser.add_argument(
        '-save_final_model',
        type = lambda x : x.lower() == 'true',
        default = "False",
        help = "최종 모델을 저장할지 여부")
        
    parser.add_argument(
        '-model_save_name',
        type = str,
        default = "BART",
        help = "최종 모델을 저장할 때 사용할 이름")

    parser.add_argument(
        '-log_dir',
        type = str,
        default = "/code/syllogistic-generation/gpt2/finetune/logs",
        help = "에폭별 훈련 로그를 저장할 위치")

    ### K-Fold Validation
    parser.add_argument(
        "-num_kfold",
        type = int,
        default = 5, # for final train, set to 0
        help = "kfold 개수")

    parser.add_argument(
        '-kfold_idx',
        type = int,
        default = 0, 
        help = "kfold 인덱스")

    if is_jupyter:
        args = parser.parse_args(args = [])
    else :
        args = parser.parse_args()
    return args
