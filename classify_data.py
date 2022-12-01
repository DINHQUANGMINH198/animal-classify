import os
import argparse
import json
import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cfg
from utils import load_json_args
from model_helper import initialize_model
import shutil



# Load a checkpoint
def predict(model, img_path, json_args):
    """ Prediction for a single test image """
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    loader = transforms.Compose([transforms.Resize(json_args['input_size']),
                                 transforms.CenterCrop(json_args['input_size']),
                                 transforms.ToTensor(),
                                 cfg.NORMALIZE
                                ])

    img = Image.open(img_path)
    img = loader(img).float()
    img = img.unsqueeze(0)

    with torch.no_grad():
        inp = img.to(torch.device(json_args['device']))
        output = model(inp)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output).numpy()[0]
        print('\nPrediction probabilities -> {}'.format(probabilities))
        _, pred = torch.max(output, 1)
        pred_category_id = pred.numpy()[0]
        pred_probability = probabilities[pred_category_id]

    model.train(mode=was_training)  # reinstate the previous mode

    return pred_category_id, pred_probability

def main():
    model_path = "models/pt_20221129_220703_1.000000.pth"

    sub_dump_dir = os.path.join(cfg.DUMP_DIR, os.path.basename(model_path)[:-13])
    json_path = os.path.join(sub_dump_dir, 'args.json')

    json_args = load_json_args(json_path)

    print(json_path)
    print(json_args)
    if json_args is None:
        return

    print("\nRUNNING ARGS:\n{}\n".format(json.dumps(json_args, indent=4)))
    
    # Initialize model
    model, params_to_update = initialize_model(is_pretrained=json_args['pretrained'])

    # Send the model to CPU or GPU
    model = model.to(torch.device(json_args['device']))

    # data path
    path_data_all = "SplitCaptcha/SplitCaptcha"
    path_data_classify = "Data_classify"


    # Setup the optimizer
    if json_args['optimizer'] == 'sgdm':
        optimizer = optim.SGD(params_to_update, lr=json_args['lr'],
                              weight_decay=json_args['weight_decay'], momentum=0.9)
    elif json_args['optimizer'] == 'adam':
        optimizer = optim.AdamW(params_to_update, lr=json_args['lr'],
                                weight_decay=json_args['weight_decay'])
    # load model                         
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(model)
    print("\nCheckpoint loaded -> epoch: {} / val loss: {:.6f} / val acc:{:.6f}".format(
        checkpoint['epoch'], checkpoint['loss'], checkpoint['acc']
    ))
    min = 1000
    for file_img in os.listdir(path_data_all):
        img_path = path_data_all + '/'+file_img
        # predict
        pred_category_id, pred_prob = predict(model, img_path, json_args)
        result_text = '{} {} ({:.2f}%)'.format(img_path ,cfg.CATEGORIES[pred_category_id], pred_prob*100)
        print('\nPrediction  ->', result_text)
        # classify using threshold (acc)
        if pred_prob*100 < min : min = pred_prob*100 
        # acc threshold
        threshold = 90 
        if pred_prob*100 >= threshold :
            data_classify_folder = path_data_classify +'/'+ cfg.CATEGORIES[pred_category_id]
            print(data_classify_folder)
            # Copy images
            try : 
                shutil.copy(os.path.join(path_data_all, file_img), os.path.join(data_classify_folder, file_img))
            except:
                 print("Co loi xay ra",file_img)
        elif  pred_prob*100 < threshold :
            data_classify_folder = path_data_classify +'/'+ "unknown"

            print(data_classify_folder)
            # Copy images
            try : 
                shutil.copy(os.path.join(path_data_all, file_img), os.path.join(data_classify_folder, file_img))
            except:
                 print("Co loi xay ra",file_img)
    print('Min ACC: ({:.2f}%)'.format(min))

    print("Classify Successful")

if __name__ == "__main__":
    main()
