
# facebook-hateful-memes-resplit
Train/dev/test split ID list for the Facebook Hateful Memes dataset, used in 'Empirical Analysis of Multimodal Fusion Techniques for Offensive Meme Classification'

# Facebook Hateful Memes (FHM) — Re-split ID Lists
This repository contains the **train / validation / test split assignment** used in our paper:

> Lalruatkimi, L., Laitonjam, L. (2026). *Empirical Analysis of Multimodal Fusion Techniques for Offensive Meme Classification.* Multimedia Tools and Applications (accepted, in press). Manuscript No. MTAP-D-25-03693.
> *(Full citation with volume/DOI will be added here once the final publication details are assigned.)*

## ⚠️ What this repository is — and is not

This repository does **not** contain any meme images or meme text. It only contains a mapping from each meme's `image_name` (the original filename used in the Facebook Hateful Memes dataset) to:

- the **label** (`0` = not-hateful, `1` = hateful) — already public in the original dataset, and
- the **split** it was assigned to in our re-split (`train`, `dev`, or `test`).

This is intentional: the Facebook Hateful Memes (FHM) dataset is distributed by Meta/Facebook AI under its own terms (research/non-commercial use), and we do not have permission to redistribute the underlying images or OCR text. What we provide here lets anyone who has separately obtained the original dataset **exactly reproduce the same splits** used in our experiments.

## Why a re-split was needed

The original FHM release (Kiela et al., 2020) contains 10,000 samples: 8,500 train, 500 dev, and 1,000 test — but the **test set is unlabeled**, which makes it unusable for standard offline evaluation and model comparison.

To enable proper, reproducible evaluation, we rebuilt the dataset as follows:

1. The unlabeled original test set (1,000 samples) was **discarded**.
2. A new labeled test set of 1,000 samples was created by **sampling from the original labeled training set** (8,500 samples), preserving class balance.
3. The remaining samples from the original training set became the new training set (7,500 samples).
4. The original dev set (500 samples, already labeled) was kept unchanged as the new dev set.

This yields **9,000 total labeled samples**, all with ground-truth labels, split as below.

### Dataset statistics (after re-splitting)

| Split | 0 (not-hateful) | 1 (hateful) | Total |
|-------|-----------------|-------------|-------|
| Train | 4,805           | 2,695       | 7,500 |
| Test  | 645             | 355         | 1,000 |
| Dev   | 250             | 250         | 500   |
| **Total** | **5,700**   | **3,300**   | **9,000** |

(These numbers match Table 6 of the paper.)

## Repository contents

```
splits/
├── train_ids.csv        # image_name, label  (7,500 rows)
├── dev_ids.csv           # image_name, label  (500 rows)
├── test_ids.csv          # image_name, label  (1,000 rows)
└── fhm_resplit_all.csv   # image_name, label, split  (all 9,000 rows combined)
scripts/
└── build_split.py        # rebuilds full train.csv/dev.csv/test.csv from your own
                           # local copy of the original dataset + the ID lists above
```

Each `image_name` corresponds exactly to the filename used in the original Facebook Hateful Memes dataset (e.g. `<img_id>.png`).

## How to reproduce our splits

1. Obtain the original Facebook Hateful Memes dataset yourself from:
   **https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset**
   (You must accept the dataset's own license/terms of use on Kaggle to download it.)
2. Extract it locally. It should contain the original labeled `train.jsonl` and `dev*.jsonl` files (or equivalent `.csv` files, for some mirrors). The original unlabeled `test.jsonl` is not needed and is not used.
3. Run the helper script from this repo:

   ```bash
   pip install pandas
   python scripts/build_split.py \
       --original_dir /path/to/your/extracted/fhm_dataset \
       --output_dir ./rebuilt_dataset
   ```

4. This writes `rebuilt_dataset/train.csv`, `dev.csv`, and `test.csv` (7,500 / 500 / 1,000 rows respectively), each with `image_name, id, text, label` — the exact split used to obtain the results reported in our paper. The `label` column is always taken from this repo's split files (authoritative); `text` and `image_name` are pulled from your local copy of the original dataset.

If the row counts or label balance in the printed summary don't match Table 6 in the paper (train: 4805/2695, dev: 250/250, test: 645/355), it usually means `--original_dir` doesn't contain the full 9,000 originally-labeled records (train + dev) — double check you passed the complete extracted dataset, not a partial download.

## Original dataset citation

If you use the Facebook Hateful Memes dataset itself, please cite the original authors:

```bibtex
@article{kiela2020hateful,
  title={The hateful memes challenge: Detecting hate speech in multimodal memes},
  author={Kiela, Douwe and Firooz, Hamed and Mohan, Aravind and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={2611--2624},
  year={2020}
}
```

## Citing this re-split / our paper

If you use this re-split in your own research, please cite our paper:

```bibtex
@article{lalruatkimi2026multimodal,
  title={Empirical Analysis of Multimodal Fusion Techniques for Offensive Meme Classification},
  author={Lalruatkimi and Laitonjam, Lenin},
  journal={Multimedia Tools and Applications},
  year={2026},
  note={In press}
}
```
*(BibTeX entry will be updated with the final volume, page numbers, and DOI once available.)*

## License

The code/files in this repository (the split-assignment lists themselves) are released under the [MIT License](LICENSE) — see the LICENSE file.

This license applies **only** to the split-list files created for this repository. It does **not** extend any rights over the original Facebook Hateful Memes dataset, which remains subject to its own license/terms as distributed by Meta/Facebook AI and the Kaggle mirror linked above. Users must independently obtain and comply with the terms of the original dataset.

## Contact

For questions about this re-split or the associated paper, please open an issue in this repository or contact:
- Lalruatkimi — ruatkimi2431@gmail.com
- Dr. Lenin Laitonjam — lenin.cse@nitmz.ac.in

Center of Excellence in Interdisciplinary Research / Department of Computer Science and Engineering,
National Institute of Technology Mizoram, Aizawl, India.
