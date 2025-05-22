from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *



def label_func(fname):
    if fname.name[0].startswith('B'):
         return 'braided'
    elif fname.name[0].startswith('S'): 
        return 'single'
    elif fname.name[0].startswith('W'): 
        return 'wandering'


path = Path('RiverImages')

fnames = get_image_files(path, folders=['Braided', 'Single', 'Wandering'])

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
print(len(failed))

dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter = RandomSplitter(),
                   item_tfms = Resize(224))
dsets = dblock.datasets(path)
dls = dblock.dataloaders(path)
# dls.show_batch()
# plt.show()

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=5)
cleaner = ImageClassifierCleaner(learn)
cleaner
plt.show()
