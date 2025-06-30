# EEG signal analysis

In this university group project, we analyzed EEG signals to differentiate between motor imagery and resting states. We used Python and scientific libraries like NumPy, SciPy, and Matplotlib for data processing, visualization, and analysis. The project aims to explore the potential of EEG in applications such as brain-computer interfaces (BCIs) and neurotechnology.

> [!NOTE]  
> We all equally contributed to this project, collaborating together on the analysis, coding, and documentation.

## Project overview

This project focuses on processing EEG data to classify brain activity under two different conditions: motor imagery (mental simulation of hand movements) and resting state (a state where the individual is not engaged in any particular mental task). We used signal processing techniques to extract key features and analyze brain patterns:

- Fourier Transforms: to analyze the frequency components of EEG signals.
- Bandpass Filtering: to isolate specific frequency bands associated with different brain states.
- Energy Analysis: to track how signal characteristics evolve over time in various brain states by applying energy analysis across multiple time windows.

## Setup and running the code

### Prerequisites

Make sure you have the following Python packages installed:

- NumPy
- SciPy
- Matplotlib

You can install these dependencies using `pip`:

```bash
pip install numpy scipy matplotlib
```

### Running the Python script

To run the analysis, simply execute the `main.py` script in your terminal or preferred Python environment:

```bash
python main.py
```

This script will load the EEG data, apply the signal processing techniques, and output visualizations of the results, including time-domain plots and energy analysis.

### LaTeX report

After running the analysis, you can view the LaTeX report (`rep.pdf`) in the `/latex` folder with all the LaTeX files.

