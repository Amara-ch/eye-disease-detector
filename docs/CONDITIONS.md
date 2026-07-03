# RFMiD - 46 Retinal Conditions

The RFMiD dataset contains **1 binary label** (Disease Risk) plus **45 disease conditions**, giving 46 label columns total.

> Note: Many conditions have very few samples (severe class imbalance). Common conditions (DR, ARMD) have hundreds of images; rare ones may have fewer than 10.

## Label List

| #  | Code | Full Name |
|----|------|-----------|
| 1  | DR   | Diabetic Retinopathy |
| 2  | ARMD | Age-Related Macular Degeneration |
| 3  | MH   | Media Haze |
| 4  | DN   | Drusen |
| 5  | MYA  | Myopia |
| 6  | BRVO | Branch Retinal Vein Occlusion |
| 7  | TSLN | Tessellation |
| 8  | ERM  | Epiretinal Membrane |
| 9  | LS   | Laser Scars |
| 10 | MS   | Macular Scar |
| 11 | CSR  | Central Serous Retinopathy |
| 12 | ODC  | Optic Disc Cupping |
| 13 | CRVO | Central Retinal Vein Occlusion |
| 14 | TV   | Tortuous Vessels |
| 15 | AH   | Asteroid Hyalosis |
| 16 | ODP  | Optic Disc Pallor |
| 17 | ODE  | Optic Disc Edema |
| 18 | ST   | Optociliary Shunt |
| 19 | AION | Anterior Ischemic Optic Neuropathy |
| 20 | PT   | Parafoveal Telangiectasia |
| 21 | RT   | Retinal Traction |
| 22 | RS   | Retinitis |
| 23 | CRS  | Chorioretinitis |
| 24 | EDN  | Exudation |
| 25 | RPEC | Retinal Pigment Epithelium Changes |
| 26 | MHL  | Macular Hole |
| 27 | RP   | Retinitis Pigmentosa |
| 28 | CWS  | Cotton Wool Spots |
| 29 | CB   | Coloboma |
| 30 | ODPM | Optic Disc Pit Maculopathy |
| 31 | PRH  | Preretinal Hemorrhage |
| 32 | MNF  | Myelinated Nerve Fibers |
| 33 | HR   | Hemorrhagic Retinopathy |
| 34 | CRAO | Central Retinal Artery Occlusion |
| 35 | TD   | Tilted Disc |
| 36 | CME  | Cystoid Macular Edema |
| 37 | PTCR | Post-Traumatic Choroidal Rupture |
| 38 | CF   | Choroidal Folds |
| 39 | VH   | Vitreous Hemorrhage |
| 40 | MCA  | Macroaneurysm |
| 41 | VS   | Vasculitis |
| 42 | BRAO | Branch Retinal Artery Occlusion |
| 43 | PLQ  | Plaque |
| 44 | HPED | Hemorrhagic Pigment Epithelial Detachment |
| 45 | CL   | Collateral |
| -  | Disease_Risk | Binary: any disease present (1) or normal (0) |

## Data Strategy

- **Stage 1:** Binary classifier -> Disease present? (uses Disease_Risk)
- **Stage 2:** Multi-label classifier -> Which of the 45 conditions?
- **Loss:** Focal / Asymmetric loss to handle extreme imbalance
- **Metric:** Macro AUC-ROC + per-label F1

> Verify exact column names against RFMiD_Training_Labels.csv when you download the dataset.
