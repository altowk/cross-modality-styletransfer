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


import numpy as np
import pyminc.volumes.factory as pyminc
from pathlib import Path
from matplotlib import pyplot as plt
import os
import scipy.signal

class RawMriVolume:
    """ Container Class representing a MRI Volume for use with the filter forest

        This class encapsules a MRI Volume in the *raw bite (unsigned)* format
        from http://brainweb.bic.mni.mcgill.ca/brainweb/about_data_formats.html .
        As the filter forest does not support processing of volumes directly, the container contains functions
        to extract single slices.
        The dimensions of the volume in each direction are specified by the sx, sy and sz attributes.
    """


    PLANE_XY = "xy"
    PLANE_XZ = "xz"
    PLANE_YZ = "yz"
    PLANES = [PLANE_YZ,PLANE_XZ,PLANE_XY]

class MincVolumeContainer:
    def __init__(self,handle):
        self.handle = handle
        self.filename = self.handle.filename
        
        
        
    def get_volume_dims(self):
        return self.handle.ndims
    
    def get_sizes(self):
        
        return [self.handle.sizes[C] for C in range(self.handle.ndims)]
    
    def _read_scaled_hyperslab(self, plane_idx, size_triple):
            #self.handle.debug = True
        
            value = self.handle.getHyperslab(plane_idx,size_triple,dtype='ushort')
            #value = np.zeros((3,3,1))
            if(np.amax(value)<=1): #this is a bug where we get wrong scaling results, so we do it manually
                value = self.handle.getHyperslab(plane_idx,size_triple,dtype='double')
                value = np.uint16(np.around(value * 65535,0))
                print(self.filename + "scaled " + self.filename + " manually, max is {}, min is {}".format(np.amax(value),np.amin(value)))
                ret = value
            else:  
                print(self.filename + "scaled by getHyperslap, max is {}, min is {}".format(np.amax(value),np.amin(value)))
                ret = value
            value = np.squeeze(value)
            return value 
    
    def get_slice_as_image(self, index: int, plane: str):
        
        '''Return a slice as image object

        Returns a single slice in one of the three planes *xy*, *xz* or *yz* as Image object which can then be
        used as input to a forest.


        Args:
            index: Index of the slice to extract. Has to be in the range of the dimension not used for the slice plane
                (e. g. [0,sz) for an xy slice).
            plane: The plane of the volume to extract the slice from

        Returns:
            Image: The selected slice as Image object

        '''
        
        if self.get_volume_dims() < 3 and plane is not RawMriVolume.PLANE_XY:
            raise Exception("cannot extract a plane other than XY from a 2 dimensional volume")

        if plane == RawMriVolume.PLANE_XY:
            width=self.handle.sizes[0]
            height=self.handle.sizes[1]
            plane_idx = (0,0,index)
            tmp=self._read_scaled_hyperslab(plane_idx,(width,height,1))
        elif plane == RawMriVolume.PLANE_XZ:
            width=self.handle.sizes[0]
            height=self.handle.sizes[2]
            plane_idx = (0,index,0)
            tmp=self._read_scaled_hyperslab(plane_idx,(width,1,height))
        elif plane == RawMriVolume.PLANE_YZ:
            width=self.handle.sizes[1]
            height=self.handle.sizes[2]
            plane_idx = (index,0,0)
            tmp=self._read_scaled_hyperslab(plane_idx,(1,width,height))
        else:
            raise Exception("invalid place for slice specified")
        
        #print(tmp)
        data= np.zeros((width,height),dtype='uint16')
        data[:,:]=tmp
        return data

def read_minc_slice(filename: str):
    volume = MincVolumeContainer(pyminc.volumeFromFile(filename))
    plane = None
    for i in range(3):
        if volume.handle.sizes[i] == 1: #we found the image plane
            plane = RawMriVolume.PLANES[i]
            break
    if plane is None:
        raise RuntimeError("could not find image plane of minc volume")
    return volume.get_slice_as_image(0,plane)

'''Reads a list of tuples containing us image and corresponding mri image cropped to the area where the us image is mostly non zero

        Args:
            mr_folder: Folder containing minc files  with one MRI image each (meaning one of the three dimensions has to be one)
            us_folder: Folder containing minc files with one US image each  matching an MRI image from mr_folder 
                       (files have to have the same filename as one image in mr_folder)
            patch_size_limit_px: If the height or width of an image pair after cropping is lower than this value, it is not returned in the result list
            
        Return:
            List[Tuple(np.array,np.array)]: List of tuples containing pairs of MRI image patches with matching ultrasound image patches


    '''
def get_non_empty_image_regions(mr_folder, us_folder, patch_size_limit_px=30):
    vol_counter = 0
    trainimages=[]
    #select some slices from the xz plane with varying noise
    for f in os.listdir(mr_orig_folder):
        if os.path.isfile(os.path.join(mr_orig_folder,f)) is False:
            continue
        mrimage = read_minc_slice(os.path.join(mr_folder,f ))
        ground_truth = read_minc_slice(os.path.join(us_folder,f))
        area_with_data = np.nonzero(ground_truth)
        if len(area_with_data[0]) < 1:
            continue
            
        #here we have the problem that the ultrasound image only occupies part of the mri image.
        #To not train with these empty areas, the following code tries to compute a rectangle as big as possible
        #containing ultrasound data. We start from the most central pixel containing data and then grow the region
        #until we encounter a spot with no data in every four directions we gan grow to
        #to compensate for single points or lines in the ultrasonic image containing no data, we do a convolution
        #with a 3 times 3 kernel to smooth these errors out and compute the region on this image 
        #(the data of the original image is returned though)

        last_x_left = last_x_right = area_with_data[0][int(len(area_with_data[0])/2)]
        last_y_top = last_y_bottom = area_with_data[1][int(len(area_with_data[1])/2)]

        converged = False
        
        
        tmpimage = np.uint16(scipy.signal.convolve2d(ground_truth,(1/9)*np.array([[1,1,1],[1,1,1],[1,1,1]])))
        tmpimage = tmpimage[1:-1,1:-1]
        test_not_converge = lambda : np.amin(tmpimage[last_x_left:last_x_right+1,last_y_top:last_y_bottom+1]) > 0

        width = ground_truth.shape[0]
        height = ground_truth.shape[1]

        while converged is False: #image still contains out of scope regions
                
            converged = True
            for direction_counter in range(4):
                if direction_counter == 0 and last_x_left > 0:
                    #tmp_mask[last_x_left-1,:]=tmp_mask[last_x_left,:]
                    last_x_left -= 1
                    if test_not_converge():
                        converged = False
                    else:
                        last_x_left += 1
                elif direction_counter == 1 and width-last_x_right > 1:
                    #tmp_mask[last_x_right+1,:]=tmp_mask[last_x_right,:]
                    last_x_right += 1
                    if test_not_converge():
                        converged = False
                    else:
                        last_x_right -= 1
                elif direction_counter == 2 and height-last_y_bottom > 1:
                    #tmp_mask[:,last_y_bottom+1]=tmp_mask[:,last_y_bottom]
                     last_y_bottom+=1
                     if test_not_converge():
                        converged = False   
                     else:
                         last_y_bottom-=1
                elif direction_counter == 3 and last_y_top > 0:
                    #tmp_mask[:,last_y_top-1]=tmp_mask[:,last_y_top]
                    last_y_top-=1
                    if test_not_converge():
                        converged = False
                    else:
                        last_y_top+=1

        
        ground_truth = ground_truth[last_x_left:last_x_right+1,last_y_top:last_y_bottom+1]
        mrimage = mrimage[last_x_left:last_x_right+1,last_y_top:last_y_bottom+1]
        
        if mrimage.shape[0] < patch_size_limit_px or mrimage.shape[1] < patch_size_limit_px:
            continue
        
        trainimages.append((mrimage,ground_truth))
        
    return trainimages

mr_orig_folder = '../tmp/mr_orig'
us_orig_folder = '../tmp/us_orig'

if __name__ == "__main__":
    print("reading slices")
    image_tuples = get_non_empty_image_regions(mr_orig_folder,us_orig_folder,30)
    print("found {} pairs of ultrasound and mir slices".format(len(image_tuples)))
    for image_tuple in image_tuples:
        plt.close("all")
        fig = plt.figure(10)
        ax = fig.add_subplot(2,2,3)
        plt.imshow(image_tuple[0],cmap='Greys_r')
        plt.title("MRI (orig)")
    
        ax = fig.add_subplot(2,2,4)
        plt.imshow(image_tuple[1],cmap='Greys_r')
        plt.title("US (orig)")
    
        plt.show()
    
       