# Enverionment
Use Anaconda to setup everything.

  -- Python 3.7

  -- PyTorch >= 1.2.0

  -- Ubuntu 18.04.2

  -- GPU: Nvidia GTX 1080Ti
  
  # To run the test code
  
1. Download the related datasets, save them to a place (=dataroot).

  -- GoPro Dataset for motion blur removal

  -- DID-MDN Dataset for rain-streak removal

  -- RESIDE-SOTS Dataset for haze removal

  -- LIVE1 for JPEG artifacts removal
  
2. Fill the scripts (test_MBN.py and test_RMBN.py) with the dataroot's path.

3. python test_MBN.py  /  python test_RMBN.py
  


