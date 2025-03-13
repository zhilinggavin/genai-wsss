# Generative-based Weakly Supervised Semantic Segmentation
This repository is under development.

## Dataset
This project includes three datasets.

### In-House - Australian IPF Registry (AIPFR) Dataset
This dataset comprises Chest HRCT scans from 227 patients and includes cases of fibrosis. No annotation or lables.

**Processed**: 
- Pre-processing pipeline: yingying completed
- Filter Selection: Only contains slices bewteen top and end slices of lung. The top and last slices are selected based on a minimum lung area of 400(to be removed after re-training of U-Net) pixels.

### In-House - OSIC Firbosis Dataset
**Annotated Source File Location**  
📂 `/media/NAS04/yyxxxx/prognostic_result/dataset/data_fibrosis/annotation_all`

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