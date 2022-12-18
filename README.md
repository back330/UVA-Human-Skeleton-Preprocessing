# UVA-Human-Skeleton-Preprocessing
The processing of the uva dataset is improved from the preprocessing method of the NTU RGB+D dataset in the CTR-GCN source code.  

UVA 3D Human Dataset address: https://github.com/SUTDCV/UAV-Human  
CTR-GCN Source code address: https://github.com/Uason-Chen/CTR-GCN

# UVA Google cloud drive address  
https://drive.google.com/drive/folders/1AgzgvLo02abnVnUQIEdATFvthWYAevt7

# Training

+ Change the config file depending on what you want.
~~~
Example: training model on UVA 3D Human cross subject with GPU 0
python main.py --config config/uav-cs-v1/default-uav.yaml --model model.ctrgcn.Model --work-dir work_dir/uav/cs-v1/ctrgcn --device 0
~~~

# Changes to get_raw_skes_data.py
1. Changing the file extension from '.skeleton' to '.txt' in 26 lines
~~~
ske_file = osp.join(skes_path, ske_name + '.txt') 
~~~
2. Changing line 29 or comment it out
~~~
print('Reading data from %s' % ske_file[-51:]) 
~~~
3. Deleting lines 43-45 from the original code
~~~
if num_bodies == 0:  # no data in this frame, drop it
     frames_drop.append(f)
     continue
~~~
4. In the following code, changing 25 to 17
~~~
joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)
~~~
5. Changing lines 51-55 to the following
~~~
for b in range(num_bodies):
    if b == 1:
          bodyID = '001'
    else:
          bodyID = '000'
    current_line += 1
    num_joints = int(str_data[current_line].strip('\r\n')) 
    current_line += 1
~~~
6. Changing lines 57-61 to: (Here we store each node coordinate with a random value between 0 and 1e-6 to eliminate the effect of all-0 data frames)
~~~
for j in range(num_joints):
    temp_str = str_data[current_line].strip('\r\n').split()
    joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32) + np.append(np.random.uniform(0, 1e-6, (1,2)), 0)
    colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
    current_line += 1
~~~
7. The 137 lines uva data set storage path was modified
~~~
skes_path = '../nturgbd_raw/nturgb+d_skeletons/'
~~~
# Changes to get_raw_denoisded_data.py
1. The threshold value of 25 lines of noise frame length is set to zero
~~~
noise_len_thres = 0
~~~
2. The filename has changed, so the following is where the label is captured
~~~
label = int(ske_name[-14:-11])
~~~
3. Changing all the numbers in the code from 25 to 17, 75 to 51, and 150 to 102
# Changes to seq_transformation.py
1. Changing all the numbers in the code to 17 for 25, 51 for 75, 102 for 150, and 34 for 50

2. In line 132, 60 is changed to 155
~~~
labels_vector = np.zeros((num_skes, 155))
~~~
3. Classify the training and testing according to the https://github.com/SUTDCV/UAV-Human amend the 197-205 lines of code is as follows (here is the reference he gave the first scheme).
~~~
train_ids = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 
             16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 
             30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 
             44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 
             59, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77,
             78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 
             102, 103, 105, 106, 110, 111, 112, 114, 115, 116, 117, 118]
test_ids = [1, 3, 4, 9, 22, 23, 24, 31, 41, 58, 60, 66, 72, 74, 75, 91, 92, 
            93, 94, 95, 96, 97, 99, 101, 104, 107, 108, 109, 113]
~~~
![image](https://github.com/back330/UVA-Human-Skeleton-Preprocessing/blob/main/Division_strategy.jpg)
# Changes to statistics
![image](https://github.com/back330/UVA-Human-Skeleton-Preprocessing/blob/main/dir.jpg)  
**The filename of all uva (A total of 23031 samples) is stored in skes_available_name.txt as shown below, and all sample action types are extracted (the number behind the blue box A in the following figure) and stored in the label.txt file as shown below:**  
![image](https://github.com/back330/UVA-Human-Skeleton-Preprocessing/blob/main/labels.jpg)   
![image](https://github.com/back330/UVA-Human-Skeleton-Preprocessing/blob/main/uva_resource.jpg)  
**Similarly, the numbers of the orange box, green box, and purple box in the above data are extracted and stored in performer.txt, setup.txt, and replication.txt files respectively, and 1 is stored in camera.txt.
Writing a program to read all the sample filenames and extract the required data into the corresponding file. The file in statistics needs to be updated before get_raw_denoisded_data.py is executed.)
Create a new updata_statistics.py with the following code:**
~~~
def updata_statistics():

    #updata label.txt
    f = open('./statistics/label.txt', "a")
    f.truncate(0)

    with open('./statistics/skes_available_name.txt', 'r') as fr:
        str_data = fr.read()
    str_data=str_data.split('\n')
    for i in range(len(str_data)):
        if i < len(str_data) - 1:
            f.write(str(int(str_data[i][-14:-11]))+'\n')
        else:
            f.write(str(int(str_data[i][-14:-11])))
    f.close()

    # updata performer.txt
    f = open('./statistics/performer.txt', "a")
    f.truncate(0)

    with open('./statistics/skes_available_name.txt', 'r') as fr:
        str_data = fr.read()
    str_data=str_data.split('\n')
    for i in range(len(str_data)):
        if i < len(str_data) - 1:
            f.write(str(int(str_data[i][1:4]))+'\n')
        else:
            f.write(str(int(str_data[i][1:4])))
    f.close()

    # updata replication.txt
    f = open('./statistics/replication.txt', "a")
    f.truncate(0)

    with open('./statistics/skes_available_name.txt', 'r') as fr:
        str_data = fr.read()
    str_data=str_data.split('\n')
    for i in range(len(str_data)):
        if i < len(str_data) - 1:
            f.write(str(int(str_data[i][-10:-9]))+'\n')
        else:
            f.write(str(int(str_data[i][-10:-9])))
    f.close()

    # updata setup.txt
    f = open('./statistics/setup.txt', "a")
    f.truncate(0)

    with open('./statistics/skes_available_name.txt', 'r') as fr:
        str_data = fr.read()
    str_data=str_data.split('\n')
    for i in range(len(str_data)):
        if i < len(str_data) - 1:
            f.write(str(int(str_data[i][5:7]))+'\n')
        else:
            f.write(str(int(str_data[i][5:7])))
    f.close()


    #updata camera.txt
    f = open('./statistics/camera.txt', "a")
    f.truncate(0)
    with open('./statistics/skes_available_name.txt', 'r') as fr:
        str_data = fr.read()
    str_data=str_data.split('\n')
    for i in range(len(str_data)):
        if i<len(str_data)-1:
            f.write(str(1)+'\n')
        else:
            f.write(str(1))
    f.close()

if __name__ == '__main__':
    updata_statistics()
~~~
# Conclude
**After making these changes, running updata_statistics.py, get_raw_skes_data.py, get_raw_denoisded_data.py, and seq_transformation.py.**
