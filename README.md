
# SGID4SE

Welcome to the SGID4SE repository! This project is part of the Wayne State University's Software Engineering and Analytics Lab (SEAL). SGID4SE stands for Sexual orientation or Gender Identity based Discrimination identification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

SGID4SE is designed to identify Sexual orientation or Gender Identity based Discriminatory content identification for Software Engineering texts. 
Find our full paper "Automated Identification of Sexual Orientation and Gender Identity Discriminatory Texts from Issue Comments" here: https://arxiv.org/pdf/2311.08485 .

Find the short paper published in Student Research Competition @ASE'22 paper titled "Identifying Sexism and Misogyny in Pull Request Comments" here: https://dl.acm.org/doi/abs/10.1145/3551349.3559515 .

## Dataset

Find our full dataset is here SGID4SE/blob/main/models/SGID-dataset-full.xlsx . It consists of 11,007 labeled GitHub pull requests and issue comments where 1,422 are identified as SGID content.

## Installation

To install SGID4SE, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/WSU-SEAL/SGID4SE.git
   ```

2. Navigate to the project directory:
   ```bash
   cd SGID4SE
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use SGID4SE, follow these instructions:

1. Run the main script to initialize the identifier system using any command from the run.sh file. For example:
   ```bash
   python SGID4SE.py --algo BERT  --ratio 1 --oversample random --bias 1 --repeat 1 --retro
   ```
Here,

- `python SGID4SE.py`: This runs the main Python script `SGID4SE.py`.
- `--algo BERT`: Specifies the algorithm to be used. In this case, BERT is selected.
- `--ratio 1`: Sets the ratio parameter to 1. This parameter controls the ratio of SGID and non-SGID content. 
- `--oversample random`: Specifies the oversampling method. Here, the `random` method is chosen, which indicates that random oversampling will be applied.
- `--bias 1`: Sets the bias parameter to 1. This parameter can be used to adjust bias for the SGID (minority) class.
- `--repeat 1`: Indicates the number of repetitions. Setting this to 1 means the process will be executed once.
- `--retro`: Enables the retro option.

### Detailed Instructions

1. **Open a Terminal or Command Prompt**:
   - On Windows, you can open Command Prompt or PowerShell.
   - On macOS or Linux, you can open Terminal.

2. **Navigate to the Project Directory**:
   Ensure you are in the directory where `SGID4SE.py` is located. You can use the `cd` command to change directories. For example:
   ```bash
   cd path/to/SGID4SE
   ```

3. **Run the Command**:
   Copy and paste the command into your terminal or command prompt, then press `Enter` to execute it:
   ```bash
   python SGID4SE.py --algo BERT --ratio 1 --oversample random --bias 1 --repeat 1 --retro
   ```

### Expected Output

Upon successful execution, the script will run using the specified parameters. Depending on the implementation of `SGID4SE.py`, you might see output related to the progress and results of the process, such as logs, analysis results, or status messages.

### Modifying Parameters

You can modify the parameters based on your requirements:

- **Algorithm**: Change `--algo BERT` to another algorithm if supported (e.g., `--algo DT`).
- **Ratio**: Adjust `--ratio 1` to another value to change the data proportion.
- **Oversampling Method**: Change `--oversample random` to another method if available (e.g., `--oversample mixed`).
- **Bias**: Set `--bias 1` to another value to adjust bias.
- **Repetitions**: Modify `--repeat 1` to run the process multiple times (e.g., `--repeat 5`).
- **Retro Mode**: The `--retro` flag can be omitted if you do not want to enable this mode.



## Contributing

We welcome contributions to the SGID4SE project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Description of your changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request describing your changes.

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or further information, please contact the project maintainers at [sayma@wayne.edu or amiangshu.bosu@wayne.edu]

---

Thank you for using SGID4SE! 
