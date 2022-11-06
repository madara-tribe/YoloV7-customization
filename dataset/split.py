import os
import shutil

label_dir = "annotations"
#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


def images_split():
    images = [os.path.join('images', x) for x in os.listdir('images')]
    annotations = [os.path.join(label_dir, x) for x in os.listdir(label_dir) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    # Move the splits into their folders
    move_files_to_folder(images, 'images/train')
    #move_files_to_folder(images, 'images/val')
    #move_files_to_folder(images, 'images/test')
    move_files_to_folder(annotations, 'annotations/train')
    #move_files_to_folder(annotations, 'annotations/val')
    #move_files_to_folder(annotations, 'annotations/test')

