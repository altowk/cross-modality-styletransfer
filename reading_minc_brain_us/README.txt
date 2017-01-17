How to generate minc data from the brainweb dataset (without any warranty):

1. The us volumes have to be aligned to the MRI volumes. An alignment transformation can be computed and saved using the register tool of the minc tool suite

2. Apply the transformation /opt/minc/bin/mincresample <us_non_registered_input.mmc> <us_registered_output.mnc> -transformation <alignment_transformation.xfm> -like <mr_volume.mnc>

3. Using generate_minc_testdata.py the now registered volume can be split up in minc files containing just one slice for each file and additionally us_images with
   displacement for training and testing are generated
   (because I did not figure out yet how to remove the third dimension from the minc file, it is still there but its width set to 1)
   => now there is one folder for the mri slices, one for the non displaced us slices and one for the displaced us slices.
   All files still contain the areas where no us image is visible
   WARNING: Old generated files are overwritten without asking. See the comment for the counter on how to create testdata for several volumes
   NOTICE: The paramters have to be adapter in the file itself

4. now these single slice files can be read in and cropped to an area where us data is avaialbe using read_minc_slices_from_file.py
   NOTICE: See function get_non_empty_image_regions and its description. __main__ contains a demo