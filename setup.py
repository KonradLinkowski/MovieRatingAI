import os
import pandas as pd

reset = input('Are you sure that you want to reset the data?(y/N): ')
if reset.lower() not in ['yes', 'y']:
    exit()


def save(data_frame, file_name):
    save_dir = 'train_data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_frame.to_csv(f'{save_dir}/{file_name}.csv', header=True, index=False)


columns = ['budget',
           'id',
           'original_language',
           'original_title',
           'popularity',
           'revenue',
           'status',
           'vote_average',
           'vote_count'
           ]
movies = pd.read_csv('dataset/movies_metadata.csv',
                     low_memory=False,
                     skipinitialspace=True,
                     usecols=columns
                     ).sample(frac=1).reset_index(drop=True)

valid_len = movies.shape[0] // 10
validation = movies.iloc[:valid_len]
test = movies.iloc[valid_len:valid_len * 2]
training = movies.iloc[valid_len * 2:]

save(training, 'train')
save(test, 'test')
save(validation, 'valid')
