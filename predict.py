from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt
import json
import argparse

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-pic", required=True, help="path to the image to classify")
ap.add_argument('-dict', required=False, default='cat_to_name.json', help="dictionary of objects to classify given as 'filename.json'")
ap.add_argument('-prt', required=False, default=False, help="print detailed model architechture")
ap.add_argument('-dev', required=False, default='cpu', help=" choose 'cpu' or 'gpu' ")
ap.add_argument('-K', required=False, default=5, help="top K classes to be returned")
args = vars(ap.parse_args())

image_path = args['pic']
if args['dev'] == 'gpu' and torch.cuda.is_available():
    device = "cuda"
elif args['dev'] == 'gpu' and torch.cuda.is_available() == False:
    print("\nNOTE: gpu selected but not available; switching to cpu")
    device = "cpu"
elif args['dev'] == 'cpu':
    device = "cpu"


def load_checkpoint(fpath):
    
    #device = "cpu"
    ##device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        fn = torch.load(fpath, map_location=str(device))
    else:
        fn = torch.load(fpath)
    classifier = nn.Sequential(nn.Linear(fn['input'], fn['hidden']),
                          nn.ReLU(),
                          nn.Dropout(p=fn['dropout']),
                          nn.Linear(fn['hidden'], fn['output']),
                          nn.LogSoftmax(dim=1))
    print(fn['arch'])
    if fn['arch'] == "vgg11":
        model = models.vgg11(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
        
    model.classifier = classifier
    model.load_state_dict(fn['state_dict'])
    #
    print("\n checkpoint data:")
    print("input", fn['input'])
    print("output", fn['output'])
    print("hidden", fn['hidden'],"\n")
    #dictionary
    class_to_idx = fn['class_to_idx']
     
    return model, class_to_idx

def process_image(image_path):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
      #  returns an Numpy array
    
 # Process a PIL image for use in a PyTorch model
    
    img = Image.open(image_path)
    print("original image size: ", img.size)
    
    w, h = img.size
    ar = w/h
    if ar>1:
        img = img.resize((int(256*ar), 256))
    else:
        img = img.resize((256, int(256*ar)))
    
    w, h = img.size
    l = (w - 224)/2
    r = (w + 224)/2
    t = (h - 224)/2
    b = (h + 224)/2
  
    img = img.crop((l,t,r,b))
    #print("processed image size", img.size)
  
    np_image = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_np_image = (np_image - mean)/std
    
    new_np_image_tr = new_np_image.transpose(2, 0, 1)
    
    proc_img = torch.from_numpy(new_np_image_tr).type(torch.FloatTensor)

    return proc_img

#def predict(image_path, model, topk=5):
def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #device = "cpu"
    print("\n my device", device)
    
    with torch.set_grad_enabled(False):
        
        if device == "cpu":
            proc_img = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor).to(device)
        else:
            proc_img = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor)
        
        output = model.forward(proc_img)            
        ps = torch.exp(output)
 
        top_p, top_class = ps.topk(topk, dim=1)
        print(top_p)
        print(top_class)
        if device == "cuda":
            top_p = top_p.cpu()
            top_class = top_class.cpu()
    
    return top_p.numpy().reshape(int(args['K'])), top_class.numpy().reshape(int(args['K']))

def dictionaries(cat_to_name, class_to_idx):
    ## Dictionaries
    #label mapping
 
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    #Invert the dictionary
    idx_to_class = {}
    for key, value in class_to_idx.items():
        idx_to_class[value] = key
    #print("Inverted dictionary \n", idx_to_class)
    
    return cat_to_name, idx_to_class

# Load checkpoint.pth
model, class_to_idx = load_checkpoint('checkpoint.pth')
#model
if args['prt'] == True:
    print(model)

# ## Class Prediction

probs, classes = predict(image_path, model, int(args['K']))
print("Probabilities and top ", args['K'], " classes")
print(probs)
print(classes)

cat_to_name, idx_to_class = dictionaries(args['dict'], class_to_idx)

#retrieve the class definitions
# loop through index to retrieve class from idx_to_class dict

print("\n Interpretation")
top_class_loc = []
top_class_def =[]
for item in classes:
    top_class_loc.append(idx_to_class[item])
    top_class_def.append(cat_to_name[idx_to_class[item]])
print(top_class_loc)
print(top_class_def)
print("\n This image is a ", top_class_def[0], " with a probability of ", probs[0],"\n")
print(" Top ", args['K'], " classes:")

n = 0
space = "      "
print("Probability   ",  "Flower name")
for item in classes:
	#print("   ", probs[n], "    ", flowers[n])
	print("  {:.3f} {} {}".format(probs[n], space, top_class_def[n]))
	n +=1

    
    


