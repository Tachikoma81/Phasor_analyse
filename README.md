---

**Library for Phasor Calculation and Visualization**

**Attributes:**
- `img`: Image matrix container.
- `lambdas`: List containing all the wavelengths.
- `phasors`: Holds the results of the method `calculate_phasor`.

**Methods:**

**Initialization:**
- **Parameters:**
    - `full_path_image`: Full path to the image.
    - `min_lambda`: Minimum wavelength.
- **Result:** Initializes your Phasor object.

**calculate_phasor:**
- **Parameters:** None.
- **Result:** Performs the phasor calculation.
- **Advice:** Use the function as follows: `pha.phasors = pha.calculate_phasor()`.

**plot_phasor:**
- **Parameters:**
    - `nb`: Number of points (default = 2000).
    - `coef`: Multiplication factor for all points (default = 2).
- **Result:** Used to plot heatmaps for the phasor.
- **Advice:** Preview the function before using this one.

**display_histogram:**
- **Parameters:** 
    - `channel` (e.g., RGB has 3 channels, starting at 0).
- **Result:** Obtains the histogram of your channel.
- **Advice:** Use the preview function (excluding `plot_phasor`) before using this one.

**hsv:**
- **Parameters:** 
    - `rgb` : boolean to change hsv image into rgb image.
- **Result:** Obtains hsv image or rgb image.
- **Advice:** Use the preview function (excluding `plot_phasor`,'display_histogram') before using this one.

**display_img:**
- **Parameters:** 
    - `img` : your phasor image 
- **Result:** display your image.
- **Advice:** Use the preview function (excluding `plot_phasor`,'display_histogram','hsv') before using this one.
---
