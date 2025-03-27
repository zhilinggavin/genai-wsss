# Generative-based Weakly Supervised Semantic Segmentation
This repository is under development.

## Dataset
This project includes three datasets.

### In-House - Australian IPF Registry (AIPFR) Dataset
This dataset comprises Chest HRCT scans from 227 patients and includes cases of fibrosis. No annotation or lables.

**Processed**: 
- Pre-processing pipeline: yingying completed
- Filter Selection: Only contains slices bewteen top and end slices of lung. The top and last slices are selected based on a minimum lung area of 400(to be removed after re-training of U-Net) pixels.
- 64495 2D slices included
    
  ```
  cd scr
  python preprocess_AIPFR.py
  ```

### In-House - OSIC Firbosis Dataset
**Annotated Source File Location**  
📂 `/media/NAS04/yyxxxx/prognostic_result/dataset/data_fibrosis/annotation_all`

**Preprocessed size350 slices File Location**  
📂 `/media/NAS04/yyxxxx/prognostic_result/dataset/data_fibrosis/gavin/slice_select`  
--> Now stored at `data/OSIC/preprocessed_size350`

- `fibrosis_selected`
- `no_fibrosis_selected`
- `no_fibrosis_covid`

**Final processed size256 slices File Location**  
📂 `data/OSIC/processed`  
processing python file: `data/OSIC/scripts/processing_fibrosis.py`
```
This script processes fibrosis images from the preprocessed OSIC dataset (size: 350x350).
The processed images will be resized to 256x256 and saved in the directory:
    data/OSIC/processed/fibrosis
```

The original fibrosis label is an annotation, which is converted to a segmentation mask through processing.
- `fibrosis_selected_gt`

**Original Slice Name Track Record**  
The processed slices are saved with randomized names in the format `orig_xxxxx.png`.  
The corresponding original names can be found in the record file: `data/OSIC/origname_record_fibrosis.csv`.

**Patient & Annotation Overview**  
- **51 patients**  
- **Annotated by 4 doctors**

| Doctor Name  | Total Cases | Case IDs |
|-------------|------------|----------------------------------------------------|
| **Tiru**    | 2          | 33, 34 |
| **Sean**    | 19         | 127-129, 131-132, 134-141, 143, 145-149 |
| **Sivandan** | 11        | 192-201, 203 |
| **Yakup**   | 21         | 64-84 |

**Processed**: 
- **51 patients**  
- **Total slices: 15,316**
  - **Annotated fibrosis slices**: 12,625  
  - **Non-fibrosis slices**: 2,691  

### Open Souce - Kits23 Dataset
contains kidney, tumour, etc... Masks for each elements.

## Results

Results are saved in the `experiments` directory under each model's folder.

**Models**: We have evaluated the performance of three segmentation models:
1. fully_supervised_unet
2. wsss_coin
3. wsss_unet

Each model has **segmentation results & quantitative results**.

- Segmentation results are saved under `/model_name/results/dataset_name/pred_mask/`.
- Quantitative results are saved under `/model_name/results/dataset_name/quant/`.

Quantitative results contain:
- 2D information for each slice
- 3D information for each case

### Project Directory Structure

The following is the directory structure of the project, showcasing the organization of files and folders:
```plaintext
my-repo/
├── .github/
│   └── workflows/
│       └── pylint.yml
├── data/
│   └── OSIC/
│       └── scripts/
│           ├── get_random_name_OSIC.py
│           └── processing_fibrosis.py
├── src/
│   └── preprocess_AIPFR.py
├── requirements.txt
└── README.md