import os
import random
import shutil
# Creating Testing and Training Folders , this step is only for one time
os.chdir(r'C:\Users\mr.geek\Desktop')
path = os.getcwd()
path = path + "\\Learning"
os.mkdir(path)
path = path + "\\Train"
os.mkdir(path)
os.mkdir(path + "\Philonthropists")
os.mkdir(path + "\Politcians")
os.mkdir(path + "\Showbiz")
os.mkdir(path + "\sportsmen")
os.mkdir(path + "\Writers")
os.chdir(r'C:\Users\mr.geek\Desktop\Learning')
path = os.getcwd()
path = path + "\\Test"
os.mkdir(path)
os.mkdir(path + "\Philonthropists")
os.mkdir(path + "\Politcians")
os.mkdir(path + "\Showbiz")
os.mkdir(path + "\sportsmen")
os.mkdir(path + "\Writers")
Total_files = 50
Training_samplese = (Total_files*80/100)
Testing_samples = (Total_files*20/100)
os.chdir(r'C:\Users\mr.geek\Desktop\preprocessing')
path = os.getcwd()
dir_names = []
paths = []
file_list = []
learn_dir_list = []
learn_paths = []
train_dir_list = []
train_paths = []
# Source DataSet paths are being stored here
for (dirpath, dirnames, filenames) in os.walk(path):
    dir_names.extend(dirnames)
path = path + "\\"
for directory in dir_names:
    paths.append(path + directory)
# # Train and Test dataset paths are being stored here
os.chdir(r'C:\Users\mr.geek\Desktop\Learning\Test')
path = os.getcwd()
for (dirpath, dirnames, filenames) in os.walk(path):
    learn_dir_list.extend(dirnames)

path = path + "\\"
for directory in dir_names:
    learn_paths.append(path + directory)
for j, k in zip(paths, learn_paths):
    i = 1
    while i <= Testing_samples:
        file = random.choice(os.listdir(j))
        shutil.move(os.path.join(j, file), os.path.join(k, file))
        i += 1
os.chdir(r'C:\Users\mr.geek\Desktop\Learning\Train')
path = os.getcwd()
for (dirpath, dirnames, filenames) in os.walk(path):
    train_dir_list.extend(dirnames)
path = path + "\\"
for directory in dir_names:
    train_paths.append(path + directory)
for j, k in zip(paths, train_paths):
    i = 1
    while i <= Training_samplese:
        file = random.choice(os.listdir(j))
        shutil.move(os.path.join(j, file), os.path.join(k, file))
        i += 1
