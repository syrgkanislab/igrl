#!/bin/bash

# Create the new image directory
mkdir -p paper_images

# List of images to copy (from previous listing)
IMAGES=(
"irae1_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_examples.png"
"irae1_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_linear_alignment.png"
"irae1_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_scatterplot.png"
"irae1_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_examples.png"
"irae1_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_linear_alignment.png"
"irae1_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_scatterplot.png"
"irae2_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_examples.png"
"irae2_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_linear_alignment.png"
"irae2_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_scatterplot.png"
"irae2_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae2_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae2_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae2_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae2_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae2_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae2_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae2_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae2_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae3_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_examples.png"
"irae3_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_linear_alignment.png"
"irae3_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_True_50_scatterplot.png"
"irae3_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae3_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae3_22_config_dgp2_conv_10_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae3_22_config_dgp2_conv_32_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae3_22_config_dgp2_conv_32_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae3_22_config_dgp2_conv_32_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae3_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae3_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae3_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"irae3_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_examples.png"
"irae3_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_linear_alignment.png"
"irae3_22_config_dgp2_dense_32_1.0_linear_hsic_50_256_5_False_50_scatterplot.png"
"justZ_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_examples.png"
"justZ_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_linear_alignment.png"
"justZ_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_scatterplot.png"
"justZ_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_examples.png"
"justZ_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_linear_alignment.png"
"justZ_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_scatterplot.png"
"vanilla_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_examples.png"
"vanilla_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_linear_alignment.png"
"vanilla_22_config_dgp1_conv_10_10.0_linear_hsic_50_256_5_scatterplot.png"
"vanilla_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_examples.png"
"vanilla_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_linear_alignment.png"
"vanilla_22_config_dgp2_dense_10_1.0_linear_hsic_50_256_5_scatterplot.png"
)

# Source directory
SRC_DIR="image_dir"

# Copy only the listed images
for img in "${IMAGES[@]}"; do

    [ -f "$SRC_DIR/$img" ] && cp "$SRC_DIR/$img" paper_images/
    # or print a warning if not found
    if [ ! -f "$SRC_DIR/$img" ]; then echo "$img not found"; fi

    echo "Copied $img"

done

# Create two subfolders for different resolutions
mkdir -p paper_images_25
mkdir -p paper_images_50

# Copy originals to both subfolders
cp paper_images/* paper_images_25/
cp paper_images/* paper_images_50/

# Resize images in each subfolder
cd paper_images_25
mogrify -resize 25% *
cd ..

cd paper_images_50
mogrify -resize 50% *
cd ..

# Zip the folders
zip -r paper_images_low_res.zip paper_images_25
zip -r paper_images.zip paper_images_50