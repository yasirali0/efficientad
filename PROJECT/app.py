import os
import time
from datetime import datetime
import streamlit as st
from PIL import Image

from loguru import logger

# from src.ml_pipeline.train_pipeline.distillation import DistillationTraining
# from src.ml_pipeline.train_pipeline.train import Trainer
from src.utils.get_config import get_config
from src.ml_pipeline.inference_pipeline.inference import Inference


infer_engine = None


# function to start the inference process
def start_inference():
    # Display the button
    st.sidebar.write('Click to start the inference process')
    start_btn = st.sidebar.button('Start Inference')
    if start_btn:

        inference_process()
            

# Function to display the inference process
def inference_process():
    
    stop_infer = st.sidebar.button('Stop Inference')

    img_placeholder = st.empty()

    while True:
        out_im_path = infer_engine.eval()
        out_im_path = next(out_im_path)
        logger.debug(f'Output Image Path: {out_im_path}')
        out_im = Image.open(out_im_path)
        img_placeholder.image(out_im, width=1100)
        time.sleep(3)

        if stop_infer:
            break


# function to choose the dataset
def choose_dataset_category():
    dataset_name = st.sidebar.selectbox('Select Dataset', options=['MVTec_AD', 'VisA'])

    if dataset_name == 'MVTec_AD':
        dataset_category = st.sidebar.selectbox('Select Category', options=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'])
    elif dataset_name == 'VisA':
        dataset_category = st.sidebar.selectbox('Select Category', options=['candle', 'capsules', 'lemon', 'cashew', 'chewinggum', 'fryum', 'nacroni1', 'macroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'])

    return dataset_name, dataset_category


# start streamlit application
if __name__ == '__main__':
    # Loading the inference api
    # api = InferenceAPI()
    # start_inference()

    from dotenv import load_dotenv
    load_dotenv()
    
    LOG_DIR = os.getenv('LOG_DIR')

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # save the logs to a file
    logger.add(f'{LOG_DIR}/{time_now}.log', rotation='10 MB', retention='10 days', level='DEBUG')

    st.set_page_config(
    page_title="EfficientAD",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
    )

    st.title('Visual Anomaly Detection')

    # make a side column for the dataset selection, and inference start button
    # and the main column for the inference process where the input and output images will be displayed
    st.sidebar.title('Menu')

    dataset_name, category = choose_dataset_category()
    # dataset_name = 'MVTec_AD'
    dataset_name, category = 'MvTec_AD', 'bottle'

    config = get_config(dataset_name)

    val_dir = config['Datasets']['eval']['root']
    model_path = config['ckpt_dir']
    result_path = config['result_dir']
    
    infer_engine = Inference(category, val_dir, model_path, result_path=result_path, ratio=1, model_size='S', device='cuda')

    st.write('The images are displayed as: Input Image | Ground Truth | Predicted Heat Map | Predicted Mask')

    start_inference()


    # distillation_training()
    # train_models(config)
    

    # if not os.path.exists(config['ckpt_dir']):
    #     os.makedirs(config['ckpt_dir'])
    # efficientad_trainer = Trainer(config=config)
    # efficientad_trainer.train(iterations=config['Model']['iterations'])


    
    