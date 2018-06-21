# SpaceTree
Reduce the high cost of manual forest monitoring using publicly available Sentinel satellite data.

Project goal is to provide an alternative to manual forest monitoring using ML system to recognise tree species, height, age, coverage etc using publicly available Sentinel satellite imagery and LVM (Latvijas Valsts Mēži) provided data. That will make valuation of the forests more accurate and faster.

# Data

## Sentinel Data

We are using Sentinel2 S2B_MSIL1C and S2A_MSIL1C data with all spectral bands for 10m and 20m spatial resolutions:
b2, b3, b4, b8, b5, b6, b7, b8a, b11, b12 and some more info on bands https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument

https://earth.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial

## LVM data

LVM provided about 500 plots of data with dominant tree spiecies and other metrics to train the model.

# Model

TBD

# Frontend

Published with GitHub pages on https://naurislv.github.io/sentilvm/

AngularJS app using CartoDB and Leaflet libraries.

To build:
ng build --prod --output-path docs --base-href "/sentilvm/"

Deploy to GitHub:
frontend/docs should be copied to root of master repository
