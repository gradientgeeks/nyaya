# train_final/ - Class Distribution Analysis

**Generated:** 2025-10-29  
**Files:** 3,873  
**Sentences:** 758,432  
**Size:** 145 MB

---

## 📊 Quick Summary

**7 Classes Total:**
- **None**: 355,669 sentences (46.90%) - Minority ✅
- **Legal Classes**: 402,763 sentences (53.10%) - **MAJORITY** ✅
- **Ratio**: None:Legal = 1:1.13

**Key Achievement:** Legal rhetorical roles now form the majority!

---

## Complete Class Distribution

| Rank | Class Name | Sentence Count | Percentage | Change from Original |
|------|------------|----------------|------------|----------------------|
| 1 | **None** | 355,669 | 46.90% | ↓ 6.81% |
| 2 | **Reasoning** | 154,716 | 20.40% | ↑ 2.37% |
| 3 | **Facts** | 135,038 | 17.80% | ↑ 2.67% |
| 4 | **Arguments of Petitioner** | 50,136 | 6.61% | ↑ 0.82% |
| 5 | **Arguments of Respondent** | 38,154 | 5.03% | ↑ 0.57% |
| 6 | **Decision** | 14,941 | 1.97% | ↑ 0.23% |
| 7 | **Issue** | 9,778 | 1.29% | ↑ 0.15% |
| **TOTAL** | | **758,432** | **100.00%** | |

---

## Legal Classes Breakdown

Within the 402,763 legal sentences:

| Class | Count | % of Total | % of Legal Content |
|-------|-------|------------|-------------------|
| **Reasoning** | 154,716 | 20.40% | 38.41% |
| **Facts** | 135,038 | 17.80% | 33.53% |
| **Arguments of Petitioner** | 50,136 | 6.61% | 12.45% |
| **Arguments of Respondent** | 38,154 | 5.03% | 9.47% |
| **Decision** | 14,941 | 1.97% | 3.71% |
| **Issue** | 9,778 | 1.29% | 2.43% |

**Top 2 classes** (Reasoning + Facts) make up **71.94%** of legal content.

---

## Comparison: Original vs Curated

### Dataset Size
| Metric | Original (train/) | Curated (train_final/) | Change |
|--------|-------------------|------------------------|--------|
| Files | 4,994 | 3,873 | -22.2% |
| Sentences | 1,123,701 | 758,432 | -32.5% |
| Size | 211 MB | 145 MB | -31.3% |

### Class Distribution Changes
| Class | Original % | Curated % | Improvement |
|-------|------------|-----------|-------------|
| None | 53.71% | 46.90% | ✅ ↓ 6.81% |
| Reasoning | 18.03% | 20.40% | ✅ ↑ 2.37% |
| Facts | 15.13% | 17.80% | ✅ ↑ 2.67% |
| Arguments of Petitioner | 5.79% | 6.61% | ✅ ↑ 0.82% |
| Arguments of Respondent | 4.46% | 5.03% | ✅ ↑ 0.57% |
| Decision | 1.74% | 1.97% | ✅ ↑ 0.23% |
| Issue | 1.14% | 1.29% | ✅ ↑ 0.15% |

**Result:** All 6 legal classes improved! ✅

### Balance Metrics
| Metric | Original | Curated | Status |
|--------|----------|---------|--------|
| None % | 53.71% | 46.90% | ✅ Reduced |
| Legal % | 46.29% | 53.10% | ✅ Increased |
| Ratio (None:Legal) | 1:0.86 | 1:1.13 | ✅ +31.4% better |
| Legal Majority? | ❌ No | ✅ Yes | ✅ **ACHIEVED** |

---

## Statistics

### Distribution Metrics
- **Average sentences per file:** 195.8
- **Largest class:** None (355,669 sentences, 46.90%)
- **Smallest class:** Issue (9,778 sentences, 1.29%)
- **Class imbalance ratio:** 36.4:1 (largest:smallest)
- **Median class size:** 50,136 sentences

### Legal Content Analysis
- **Total legal sentences:** 402,763 (53.10%)
- **Reasoning contribution:** 38.41% of legal content
- **Facts contribution:** 33.53% of legal content
- **Arguments contribution:** 21.92% of legal content
- **Decision + Issue contribution:** 6.14% of legal content

---

## Recommended Training Configuration

### Class Weights (Inverse Frequency)

```python
# Use these weights to handle remaining imbalance
class_weights = {
    'None':                      0.40,  # Most frequent (46.90%)
    'Reasoning':                 0.92,
    'Facts':                     1.05,
    'Arguments of Petitioner':   2.84,
    'Arguments of Respondent':   3.71,
    'Decision':                  9.54,  
    'Issue':                    15.23   # Least frequent (1.29%)
}
```

### Training Recommendations

1. **Dataset Path:** Use `dataset/train_final/` for training
2. **Class Weighting:** Apply weights above for balanced learning
3. **Metrics:** Track macro-F1 and per-class F1 (not just accuracy)
4. **Validation:** Use original `dataset/val/` (unchanged)
5. **Testing:** Use original `dataset/test/` (unchanged)

### Expected Benefits

✅ **Faster convergence** - 32.5% fewer sentences per epoch  
✅ **Better legal recall** - Legal classes now majority  
✅ **Reduced 'None' bias** - No longer dominant class  
✅ **Improved macro-F1** - Balanced class distribution  
✅ **Better generalization** - Diverse file composition

---

## Quality Assessment

### ✅ Strengths
1. **Balanced:** Legal classes (53.10%) > None (46.90%)
2. **Complete:** All 7 classes present
3. **Improved:** All 6 legal classes boosted
4. **Efficient:** 32.5% smaller, faster training
5. **Diverse:** Mix of legal-heavy and none-heavy files

### ⚠️ Remaining Challenges
1. **Class imbalance:** Issue (1.29%) and Decision (1.97%) still small
2. **Imbalance ratio:** 36.4:1 between largest and smallest
3. **Mitigation:** Use class weights (provided above)

---

## Visual Distribution

```
None                 ████████████████████████ 46.90%
Reasoning            ██████████ 20.40%
Facts                ████████ 17.80%
Args Petitioner      ███ 6.61%
Args Respondent      ██ 5.03%
Decision             █ 1.97%
Issue                █ 1.29%
```

**Legal Classes Combined:** ████████████████████████████ 53.10% ✅

---

## File Composition

The 3,873 files consist of:
- **2,983 legal-heavy files** (76.8%) - where None < Other classes
- **900 none-heavy files** (23.2%) - randomly selected for diversity

This composition ensures:
- Strong signal for learning legal roles
- Prevents overfitting to highly structured documents
- Maintains real-world variance

---

## Comparison to Standard Datasets

| Characteristic | train_final/ | Typical NLP Dataset |
|----------------|--------------|---------------------|
| Class balance | 46.90% vs 53.10% | Often 90%+ majority |
| Improvement from original | ↑ 6.81% | N/A |
| Legal majority achieved | ✅ Yes | N/A |
| Domain-specific | Legal judgments | General text |
| Fine-tuning ready | ✅ Yes | Varies |

**Conclusion:** This dataset is well-balanced for legal domain fine-tuning.

---

## Next Steps

1. ✅ **Dataset ready:** Use `train_final/` for training
2. ✅ **Class weights prepared:** Apply weights from above
3. ✅ **Documentation complete:** See README.md for full details
4. 🚀 **Start training:** Fine-tune InLegalBERT on this curated dataset

---

**See also:**
- `README.md` - Complete dataset documentation
- `FILE_LIST.txt` - Provenance of all files
- `../CURATION_SUMMARY.md` - Curation process details
- `../TRAINING_DATASET_ANALYSIS.md` - Original dataset analysis
