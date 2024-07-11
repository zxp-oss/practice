"Image" folders contain GF-2 images, "Annotation" folders contain label masks, and "Coordinate" folder provides the corresponding coordinate files.

Image_16bit_BGRNir: 150 raw 16-bit GF-2 images, of which the band order is B, G, R, Nir.
Image__8bit_NirRGB: 150 re-quantized 8-bit GF-2 images, of which the band order is Nir, R, G, B.

Annotation__color: 150 colorful label masks, with colors representing different categories.
Annotation__index: 150 single-channel label masks, with pixel values representing class indexes.

* Different re-quantization methods have an impact on the performance of domain adaptation, so we provide both raw and processed (used in our article) images. Raw 16-bit images can be used more flexibly for a variety of potential applications. For 8-bit images, we change the band order to make it easier to annotate. The annotation information of color label and index label are the same. The colorful label masks can be used for visual observation, and the indexed label masks can be used directly for training.

* Correspondence of colors (RGB) and categories:
200,   0,   0: industrial area
  0, 200,   0: paddy field
150, 250,   0: irrigated field
150, 200, 150: dry cropland
200,   0, 200: garden land
150,   0, 250: arbor forest
150, 150, 250: shrub forest
200, 150, 200: park
250, 200,   0: natural meadow
200, 200,   0: artificial meadow
  0,   0, 200: river
250,   0, 150: urban residential
  0, 150, 200: lake
  0, 200, 250: pond
150, 200, 250: fish pond
250, 250, 250: snow
200, 200, 200: bareland
200, 150, 150: rural residential
250, 200, 150: stadium
150, 150,   0: square
250, 150, 150: road
250, 150,   0: overpass
250, 200, 250: railway station
200, 150,   0: airport
  0,   0,   0: unlabeled

* Correspondence of indexes and categories:
 1: industrial area
 2: paddy field
 3: irrigated field
 4: dry cropland
 5: garden land
 6: arbor forest
 7: shrub forest
 8: park
 9: natural meadow
10: artificial meadow
11: river
12: urban residential
13: lake
14: pond
15: fish pond
16: snow
17: bareland
18: rural residential
19: stadium
20: square
21: road
22: overpass
23: railway station
24: airport
 0: unlabeled

* Test images for baseline (Table 4 in our article):
GF2_PMS1__L1A0001064454-MSS1.tif
GF2_PMS1__L1A0001118839-MSS1.tif
GF2_PMS1__L1A0001344822-MSS1.tif
GF2_PMS1__L1A0001348919-MSS1.tif
GF2_PMS1__L1A0001366278-MSS1.tif
GF2_PMS1__L1A0001366284-MSS1.tif
GF2_PMS1__L1A0001395956-MSS1.tif
GF2_PMS1__L1A0001432972-MSS1.tif
GF2_PMS1__L1A0001670888-MSS1.tif
GF2_PMS1__L1A0001680857-MSS1.tif
GF2_PMS1__L1A0001680858-MSS1.tif
GF2_PMS1__L1A0001757429-MSS1.tif
GF2_PMS1__L1A0001765574-MSS1.tif
GF2_PMS2__L1A0000607677-MSS2.tif
GF2_PMS2__L1A0000607681-MSS2.tif
GF2_PMS2__L1A0000718813-MSS2.tif
GF2_PMS2__L1A0001038935-MSS2.tif
GF2_PMS2__L1A0001038936-MSS2.tif
GF2_PMS2__L1A0001119060-MSS2.tif
GF2_PMS2__L1A0001367840-MSS2.tif
GF2_PMS2__L1A0001378491-MSS2.tif
GF2_PMS2__L1A0001378501-MSS2.tif
GF2_PMS2__L1A0001396036-MSS2.tif
GF2_PMS2__L1A0001396037-MSS2.tif
GF2_PMS2__L1A0001416129-MSS2.tif
GF2_PMS2__L1A0001471436-MSS2.tif
GF2_PMS2__L1A0001517494-MSS2.tif
GF2_PMS2__L1A0001591676-MSS2.tif
GF2_PMS2__L1A0001787564-MSS2.tif
GF2_PMS2__L1A0001821754-MSS2.tif