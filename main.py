import os
import logging
import argparse
import configparser
from src.modules import EvaluateProcess, SubmitProcess

logging.basicConfig(level=logging.INFO)

def main(args):

    logging.info('讀取設定檔.')

    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # NOTE read config file
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** raw_data
    train_path = config_client.get('raw_data', 'train')
    test_path = config_client.get('raw_data', 'test')

    # ** task
    task = config_client.get('task', 'task')
    preprocess_version = config_client.get('task', 'preprocess_version')
    estimator_version = config_client.get('task', 'estimator_version')

    # ** submit
    output_path = config_client.get('submit', 'output_path')

    # ** evaluate
    cv = config_client.getint('evaluate', 'cv')
    load_path = config_client.get('evaluate', 'load_path')
    save_path = config_client.get('evaluate', 'save_path')

    if task == 'evaluate':
        evaluate_process = EvaluateProcess(train_path, preprocess_version, estimator_version, cv, save_path, load_path)
        evaluate_process.run()
    elif task == 'submit':
        submit_process = SubmitProcess(train_path, test_path, preprocess_version, estimator_version, output_path)
        submit_process.run()
    else:
        raise NotImplementedError('無效的task')
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')   

    args = argparser.parse_args()
    main(args)

