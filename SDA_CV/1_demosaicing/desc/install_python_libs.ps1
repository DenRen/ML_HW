..\venv\Scripts\activate
python -V

function InstallLibs() {
    python -m pip install      `
        contourpy==1.1.1       `
        cycler==0.11.0         `
        exceptiongroup==1.1.3  `
        fonttools==4.42.1      `
        imageio==2.31.3        `
        iniconfig==2.0.0       `
        joblib==1.3.2          `
        kiwisolver==1.4.5      `
        lazy_loader==0.3       `
        matplotlib==3.8.0      `
        networkx==3.1          `
        numpy==1.26.0          `
        packaging==23.1        `
        Pillow==10.0.1         `
        pluggy==1.3.0          `
        pyparsing==3.1.1       `
        pytest==7.4.2          `
        python-dateutil==2.8.2 `
        PyWavelets==1.4.1      `
        scikit-image==0.21.0   `
        scikit-learn==1.3.0    `
        scipy==1.11.2          `
        six==1.16.0            `
        threadpoolctl==3.2.0   `
        tifffile==2023.8.30    `
        tomli==2.0.1
}

InstallLibs