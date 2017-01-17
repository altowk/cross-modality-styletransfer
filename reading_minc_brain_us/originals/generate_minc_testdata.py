__author__ = 'Benjamin Schnoy'
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import os

minc_resample_executable = "/opt/minc/bin/mincresample"

#input data paramters, mri_in has to be the original mri volume from the dataset, 
# us_in the aligned ultrasound volume from the dataset resampled to have the same dimensions has the mri volume
input_data_folder = "../input_data/06/"
mri_in = "06_mr_tal.mnc"
us_in = "us_registered.mnc"
start_coords = [-72,-134,-98] #start coordinates in the registered volumes, adjust if necessary


#output parameters
output_data_folder = "../tmp/" #the folder for the output data
transformation_out = output_data_folder + "transformations/" # the folder the transformations from us_aligned to us_transformed are stored
mri_original_out = output_data_folder + "mr_orig/" # folder containing the mri slices
us_aligned_out = output_data_folder + "us_orig/" # folder containing the aligned us slices
us_transformed_out = output_data_folder + "us_displaced/"  # folder containing the displaced us slices

#settings for displacing the ultrasonic images
max_displacement = 10
max_rot = 15
max_scale = 1.2
min_scale = 0.8

import numpy as np
import pyminc.volumes.factory as pyminc
import subprocess



(AXIS_X,AXIS_Y,AXIS_Z) = range(3)
plane_names = ['yz','xz','xz']


transformation_string_template_begin = \
"""MNI Transform File
%VIO_Volume: 01_mr_tal.mnc
%VIO_Volume: 01a_us_tal.mnc

Transform_Type = Linear;
Linear_Transform =
 {} {} {} {}
 {} {} {} {}
 {} {} {} {};"""

def ranged_symmetric_random_sample(abs_max):
    return np.random.ranf() * 2 * abs_max - abs_max
    


def create_2d_random_transform(orthogonal_axis):
    matrix = np.zeros((3,4))
    matrix[0,0] = matrix[1,1] = matrix[2,2] = 1
    displacements = [ranged_symmetric_random_sample(max_displacement),ranged_symmetric_random_sample(max_displacement)] #= [-max_displacement,max_displacement)
    rotation = [ranged_symmetric_random_sample(max_rot)/360*2*np.pi]
    scales = [(max_scale-min_scale) * np.random.ranf()+min_scale,(max_scale-min_scale) * np.random.ranf()+min_scale]
    ind = 0
    for i in range(3):
        if i == orthogonal_axis:
            continue
            
        matrix[i,i]=scales[ind]*np.cos(rotation)
        if ind == 0:
            offset = 2 if orthogonal_axis == AXIS_Y else 1 
            matrix[i+offset,i]=scales[ind]*(-1)*np.sin(rotation)
            matrix[i,i+offset]=scales[ind+1]*np.sin(rotation)
            matrix[i,3]=displacements[ind]*scales[ind]*np.cos(rotation)-displacements[1]*scales[1]*np.sin(rotation)
        else:
            matrix[i,3]=displacements[ind]*scales[ind]*np.sin(rotation)+displacements[1]*scales[1]*np.cos(rotation)
        ind+=1
    return transformation_string_template_begin.format(*(matrix.flatten()))

if __name__ == "__main__":
    mri_volume = pyminc.volumeFromFile(input_data_folder + '/' + mri_in) 
    print(str(mri_volume.sizes[0]) + '/' + str(mri_volume.sizes[1]) + '/' + str(mri_volume.sizes[2]) )
    counter = 0 #if we have an offset from generating files from other minc volumes we have to set this value to the index to start  from
               #to no override files
    
    for folder in [us_transformed_out,us_aligned_out,transformation_out,mri_original_out]:       
        if not os.path.isdir(folder):
            print("Creating dir {}".format(os.path.abspath(us_transformed_out)))
            os.mkdir(folder)
        else:
            print("folder {} exists".format((us_transformed_out)))
            
        
    for i in range(3):
        for j in range(start_coords[i],mri_volume.sizes[i]+start_coords[i],8):
            
            transform_matrix = create_2d_random_transform(i)
                
            #transform_filename = transformation_out + us_in + "_plane_" + plane_names[i] + "pos_" + str(j) + ".xfm"
            transform_filename = transformation_out + str(counter) + ".xfm"
            f = open(transform_filename,"w")
            f.write(transform_matrix)
            f.close()
            us_infilename = input_data_folder + us_in
            mr_infilename = input_data_folder + mri_in
            us_transformed_file = us_transformed_out + str(counter) + ".mnc"
            us_original_file = us_aligned_out + '/' + str(counter) + ".mnc"
            mr_original_file = mri_original_out + '/' + str(counter) + ".mnc"
            number_elements = np.zeros((3),dtype=int)
            number_elements[i] = 1
            local_start = start_coords.copy()
            local_start[i] = j
            
            
            
            #this generates the displaced us minc file
            cmdline = minc_resample_executable + ' ' + us_infilename + " " + us_transformed_file +  \
            " -transformation " + transform_filename + " -like " +  us_infilename + " " + "-clobber -nelements {} {} {}".format(*number_elements) + " " + \
            "-start {} {} {}".format(*local_start)
            print(cmdline)
            subprocess.run(cmdline,shell=True,check=True)
            
            #this generates the original us minc file
            cmdline = minc_resample_executable + ' ' + us_infilename + " " + us_original_file +  \
            " -like " +  us_infilename + " " + "-clobber -nelements {} {} {}".format(*number_elements) + " " + \
            "-start {} {} {}".format(*local_start)
            print(cmdline)
            subprocess.run(cmdline,shell=True,check=True)
            
               #this generates the original mr minc file
            cmdline = minc_resample_executable + ' ' + mr_infilename + " " + mr_original_file +  \
            " -like " +  mr_infilename + " " + "-clobber -nelements {} {} {}".format(*number_elements) + " " + \
            "-start {} {} {}".format(*local_start)
            print(cmdline)
            subprocess.run(cmdline,shell=True,check=True)
            
            us_trans_volume = pyminc.volumeFromFile(us_transformed_file)
            mr_volume = pyminc.volumeFromFile(mr_original_file)
            if us_trans_volume.data.max() < 0.1:
                print("Skipping empty us volume")
                continue
            
            counter+=1
    #    f.close()
            
            
    #for i in range(3):
    #    matrix = create_2d_random_transform(i)
    #    filename = transformation_out + "test_axis_" + str(i) + ".xfm"
    #    f = open(filename,"w")
    #    f.write(matrix)
    #    f.close()
    #    print("Wrote " + filename)
    
    
    
    
        
    