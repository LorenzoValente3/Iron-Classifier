# Iron-Classifier

Iron-Classifier is a project that focuses on classifying the Iron spectrum from the light nuclei in a supervised setting using various neural network model approaches. 
The project utilizes data sourced from the private Monte Carlo data for the MAGIC collaboration, specifically for heavy nuclei in cosmic rays.

## Project Structure

* [ad](./ad): The main namespace of the project.
  *  [models](./ad/models): Contains custom models for classification.
  *  [utils](./ad/utils.py): Provides general plots and utility code.
  *  [plots](./ad/plots.py): Includes additional code for generating plots related to the classification and performance evaluation.

* [weights](./weights): contains the pre-trained weights of the various models.

The `ad` namespace serves as the core module where the different models for classification are defined. The `models` subpackage contains custom implementations of neural network models optimized for the specific task of classifying Iron spectra.
Additionally, the `utils.py` module houses general plots and utility functions essential for data preprocessing, evaluation, and visualization. 
To further enhance the project, the `plots.py` file has been introduced to the structure, enabling the generation of additional plots and visualizations related to the classification process.

Feel free to modify, adjust, and expand the readme according to your specific needs and the details of your project.
