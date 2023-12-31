// converts an RBG image to grayscale
__kernel void rgbToGray(
    __global unsigned char* inImg,
    __global unsigned char* outImg,
    const int bytesPerPix)
{
    // Get work-item identifiers.
    int x = get_global_id(0);
    int y = get_global_id(1);
    int imgW = get_global_size(0);

    int offset = ((y * imgW) + x) * bytesPerPix;
    unsigned char B = inImg[offset];
    unsigned char G = inImg[offset + 1];
    unsigned char R = inImg[offset + 2];
    unsigned char gray = (B + G + R) / 3;
    
    outImg[offset] = gray;
    outImg[offset + 1] = gray;
    outImg[offset + 2] = gray;
    if (bytesPerPix == 4) // if we have alfa-channel
        outImg[offset + 3] = inImg[offset + 3];
}

int MinMaxVal(int min, int max, int val)
{
    if (val < min) 
    {
        return min;
    }
    else if (val > max) 
    {
        return max;
    }
    return val;
}

// This kernel function convolves an image input_image[imgWidth, imgHeight]
// with a mask of size maskSize.
__kernel void filterImage(
    __global unsigned char* inImg,
    __global unsigned char* outImg,
    const int bytesPerPix,
    const unsigned int maskSize,
    __constant float* mask)
{
    // Get work-item identifiers.
    int x = get_global_id(0);
    int y = get_global_id(1);
    int imgW = get_global_size(0);
    int imgH = get_global_size(1);
    int offset = ((y * imgW) + x) * bytesPerPix;

    // Check if the mask cannot be applied to the current pixel
    if (x < maskSize / 2
        || y < maskSize / 2
        || x >= imgW - maskSize / 2
        || y >= imgH - maskSize / 2)
    {
        outImg[offset] = 0;
        outImg[offset + 1] = 0;
        outImg[offset + 2] = 0;
        if (bytesPerPix == 4) // if we have alfa-channel
            outImg[offset + 3] = inImg[offset + 3];
        return;
    }

    // Apply mask based on the neighborhood of pixel inputImg.
    int outSumB = 0;
    int outSumG = 0;
    int outSumR = 0;
    for (size_t k = 0; k < maskSize; k++)
    {
        for (size_t l = 0; l < maskSize; l++)
        {
            // Calculate the current mask index.
            size_t maskIdx = (maskSize - 1 - k) + (maskSize - 1 - l) * maskSize;
            // Compute output pixel.
            size_t xM = x - maskSize / 2 + k;
            size_t yM = y - maskSize / 2 + l;
            int offsetM = ((yM * imgW) + xM) * bytesPerPix;
            outSumB += inImg[offsetM] * mask[maskIdx];
            outSumG += inImg[offsetM + 1] * mask[maskIdx];
            outSumR += inImg[offsetM + 2] * mask[maskIdx];
        }
    }

    // Write output pixel.
     outImg[offset] = MinMaxVal(0, 255, outSumB);
     outImg[offset + 1] = MinMaxVal(0, 255, outSumG);
     outImg[offset + 2] = MinMaxVal(0, 255, outSumR);
    if (bytesPerPix == 4) // if we have alfa-channel
        outImg[offset + 3] = inImg[offset + 3];
}

// This kernel function convolves an image input_image[imgWidth, imgHeight]
// with a mask of size maskSize by caching submatrices from the input image 
// in the device local memory.
__kernel void filterImageCached(
    __global unsigned char* inImg,
    __global unsigned char* outImg,
    const int bytesPerPix,
    const unsigned int maskSize,
    __constant float* mask)
{
    // Get work-item identifiers.
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int imgW = get_global_size(0);
    int imgH = get_global_size(1);
    int offset = ((y * imgW) + x) * bytesPerPix;

    // Declare submatrix used to cache the input image on local memory.
    __local unsigned char sub[SUB_SIZE][SUB_SIZE][4];

    sub[ly][lx][0] = inImg[offset];
    sub[ly][lx][1] = inImg[offset + 1];
    sub[ly][lx][2] = inImg[offset + 2];
    if (bytesPerPix == 4) // if we have alfa-channel
        sub[ly][lx][3] = inImg[offset + 3];

    // Synchronize all work-items in this work-group.
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Check if the mask cannot be applied to the current pixel
    if (x < maskSize / 2
        || y < maskSize / 2
        || x >= imgW - maskSize / 2
        || y >= imgH - maskSize / 2)
    {
        outImg[offset] = 0;
        outImg[offset + 1] = 0;
        outImg[offset + 2] = 0;
        if (bytesPerPix == 4) // if we have alfa-channel
            outImg[offset + 3] = inImg[offset + 3];
        return;
    }

    // Apply mask based on the neighborhood of pixel inputImg.
    int outSumB = 0;
    int outSumG = 0;
    int outSumR = 0;
    for (size_t k = 0; k < maskSize; k++)
    {
        for (size_t l = 0; l < maskSize; l++)
        {
            // Calculate the current mask index.
            size_t maskIdx = (maskSize - 1 - k) + (maskSize - 1 - l) * maskSize;
            // Compute output pixel.
            size_t maskLX = lx - maskSize / 2 + k;
            size_t maskLY = ly - maskSize / 2 + l;
            // �heck if the current input pixel is in the local memory 
            if (maskLX >= 0 && maskLX < SUB_SIZE && maskLY >= 0 && maskLY < SUB_SIZE)
            {
                outSumB += sub[maskLY][maskLX][0] * mask[maskIdx];
                outSumG += sub[maskLY][maskLX][1] * mask[maskIdx];
                outSumR += sub[maskLY][maskLX][2] * mask[maskIdx];
            }
            else
            {
                // Read the current input pixel from the global memory
                size_t maskX = x - maskSize / 2 + k;
                size_t maskY = y - maskSize / 2 + l;
                int offsetM = ((maskY * imgW) + maskX) * bytesPerPix;
                outSumB += inImg[offsetM] * mask[maskIdx];
                outSumG += inImg[offsetM + 1] * mask[maskIdx];
                outSumR += inImg[offsetM + 2] * mask[maskIdx];
            }
        }
    }

    // Write output pixel.
    outImg[offset] = MinMaxVal(0, 255, outSumB);
    outImg[offset + 1] = MinMaxVal(0, 255, outSumG);
    outImg[offset + 2] = MinMaxVal(0, 255, outSumR);
    if (bytesPerPix == 4) // if we have alfa-channel
        outImg[offset + 3] = inImg[offset + 3];
}