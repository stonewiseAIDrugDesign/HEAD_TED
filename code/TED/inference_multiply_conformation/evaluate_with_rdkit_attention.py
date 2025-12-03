import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow_addons.layers import MultiHeadAttention
import logging

logger = logging.getLogger('Log')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(predict_name: str, base_local_path):
    model_file = f'{base_local_path}/base_model_with_xtb_dft_finetuning.h5'
    model = tf.keras.models.load_model(model_file, custom_objects={'Addons>MultiHeadAttention': MultiHeadAttention})
    # model = tf.keras.models.load_model(model_file)
    valid_file_size = 8
    start_file_id = 8
    test_data = np.load(f'{base_local_path}/rdkit_base_dihedral/{predict_name}_X_test.npy'
                        f'')
    test_id_data = np.load(f'{base_local_path}/rdkit_base_dihedral/{predict_name}_y_test.npy')

    with open(f'{base_local_path}/train_valid_scale.pkl', 'rb') as fptr:
        scaler = pickle.load(fptr)
    test_data = scaler.transform(test_data)
    test_data = test_data.reshape(int(test_data.shape[0] / 24), 24, 293)
    logger.info(f'feature data:{test_data.shape},test_id:{test_id_data.shape}')
    logger.info(f'model info is:{model.summary()}')
    # mtl_pred = model.predict(test_data)
    # y_pred = mtl_pred[1]
    # logger.info(f'y_mtl_pred:{y_pred.shape}')
    # y_pred = np.reshape(y_pred, (-1, y_pred.shape[1]))
    # logger.info(f'y_new_pred:{y_pred.shape}')
    y_pred = model.predict(test_data)
    logger.info(f'y_pred:{y_pred.shape}')
    y_pred_min = np.min(y_pred, axis=1)
    y_pred_min = y_pred_min.reshape(y_pred_min.shape[0], 1)
    y_pred = y_pred - y_pred_min
    y_pred = y_pred.flatten()
    y_pred = y_pred.reshape(y_pred.shape[0], 1)
    logger.info(f'new y_pred is:{y_pred.shape}')
    test_id_data = test_id_data[:, 1]
    test_id_data = test_id_data.reshape(test_id_data.shape[0], 1)
    ret_y_pred = np.concatenate((y_pred, test_id_data), axis=1)
    ret_file_path = f'{base_local_path}/rdkit_base_dihedral/{predict_name}_model_predict.npy'
    np.save(ret_file_path, ret_y_pred)
    logger.info(f'return value is:{ret_y_pred.shape}')
    return ret_y_pred