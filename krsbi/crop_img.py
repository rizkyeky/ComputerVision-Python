import pandas as pd
from PIL import Image

train_labels = pd.read_csv('train_labels.csv')
val_labels = pd.read_csv('val_labels.csv')
test_labels = pd.read_csv('test_labels.csv')

labels = train_labels.append([val_labels, test_labels])

for i, row in labels.iterrows():
    try:
        img = Image.open('data/'+row['filename'])
        img = img.convert('RGB')
        img = img.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax']))

        try:
            filename = row['class']+str(i)+row['filename'][:5]
        except:
            filename = row['class']+str(i)
        img.save('crop/'+row['class']+'/'+filename+'.jpg')
    except:
        print('Error:', row['filename'])