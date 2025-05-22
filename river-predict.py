from fastai.vision.all import *
import tkinter as tk
from tkinter import filedialog

#### Script to predict loaded images using the trained model ####
learn = load_learner('export.pkl')

categories = ('braided', 'single', 'wandering')
    
## PREDICT IMAGE
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return pred
    # return dict(zip(categories, map(float, probs)))

# create a root window
root = tk.Tk()
root.withdraw()

# open the file dialog box
file_path = filedialog.askdirectory()

# iterate over files in
# that directory
# to create a list for batch prediction
files = []
filenames = []
for filename in os.listdir(file_path):
    if filename.lower().endswith('.png'):
        f = os.path.join(file_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
           files.append(f)
           filenames.append(filename)

# Predict batch
batch_dl = learn.dls.test_dl(files)
preds, _, decoded = learn.get_preds(dl=batch_dl, with_decoded=True)
predictions = []
for d in decoded:
    predictions.append(categories[d.item()])

result = ''

for i in range(len(predictions)):
    result = result + filenames[i] + ',' + predictions[i] + '\n'

# Save predictions to file
f = open('predictions.csv','w')
f.write(result) #Give your csv text here.
## Python will convert \n to os.linesep
f.close()




    