# The Project Explained Simply

**Amazon Inventory Reconciliation Using AI** - Aarush Patil (PES1UG24AM005) and Aryan Arora (PES1UG24AM055)

---

## 1. The problem in one minute

Amazon stores products in fabric bins in giant warehouses. Sometimes an item gets put in the wrong bin, so:

- The **computer says** "this bin has 3 items".
- The **bin really has** 2 items.

That mismatch is called an **inventory variance**. Checking every bin by hand is impossible, so we ask: *can a computer look at a photo of the bin, count the items, and shout when the count disagrees with the system?*

That is the whole project: **photo in, item count out, compare with the record, flag mismatches.**

---

## 2. The data

- **Dataset:** Amazon Bin Image Dataset (ABID), a free public set of 535,234 real bin photos. Each photo comes with a small file that says how many items are in the bin. That number is the **label** (the "right answer").
- **What we used:** a random sample of **30,000 photos** of bins that hold **0 to 5 items**.
- **Split:** 21,000 for learning (train), 6,000 for tuning choices (validation), 3,000 for the final exam (test). The test photos are never used for learning.
- **Class sizes:** 0 items: 848, 1: 3,368, 2: 6,387, 3: 7,598, 4: 6,660, 5: 5,139. Bins with 3 items are the most common.
- **Why it is hard:** items are covered by tape or plastic, piled on each other, or all different shapes. Some photos are so unclear that even a person cannot count them.

We treat it as a **6-way multiple choice**: is the answer 0, 1, 2, 3, 4 or 5?

---

## 3. The pipeline (what happens, in order)

| Step | Script | In plain words |
|---|---|---|
| 1 | `download_data.py` | Fetch 30,000 random photos, shrink each to 224x224 pixels |
| 2 | `build_splits.py` | Divide them into train / validation / test |
| 3 | `baselines.py` | Try the "old-school" methods first |
| 4 | `train_cnn.py` | Train the deep-learning model (twice) |
| 5 | `make_results.py` | Compute scores, draw charts, test the mismatch detector |
| 6 | `demo.py` | Upload a photo + type the recorded count, get MATCH or VARIANCE |

---

## 4. Method 1: the old-school baselines

We need something to compare against, so we first tried classic machine learning.

1. Shrink each photo to 64x64 and turn it into a long list of numbers (pixel colours).
2. **PCA** squeezes that huge list down to 256 numbers that keep most of the information (about 93%). Think of summarising a long book into its key points.
3. Feed the 256 numbers to **Logistic Regression, Linear SVM, Decision Tree, RBF-SVM**.
4. Also try **HOG** (a trick that describes the directions of edges in the image) with an RBF-SVM.

These models only see numbers, not shapes, so they struggle with messy bins.

---

## 5. Method 2: the deep-learning model (the main one)

### ResNet34 and transfer learning
A **CNN** (convolutional neural network) looks at images in layers: first edges, then shapes, then whole objects. **ResNet34** is a famous 34-layer CNN already trained on millions of everyday photos (ImageNet).

**Transfer learning** = instead of teaching a model to see from zero, we start from one that already knows how to see, and teach it our specific job. Like hiring an experienced photographer and teaching them to count items, instead of training a baby.

### How we trained it (two stages)
1. **Stage 1 (2 epochs):** keep the old knowledge frozen, train only the new final layer that outputs "0 to 5".
2. **Stage 2 (9 epochs):** unfreeze everything and gently fine-tune the whole network.

### The tricks, in simple words
- **Discriminative learning rates:** early layers know basic things (edges), so they change very slowly (lr/9); middle layers change a bit more (lr/3); the new top layer changes the most (lr).
- **SGDR (cosine warm restarts):** the learning rate shrinks smoothly, then jumps back up and shrinks again. The jump helps the model escape bad spots.
- **LR finder:** a quick test that tries many learning rates and picks a good one automatically.
- **Data augmentation:** randomly flip photos and tweak colours so the model sees more variety.
- **Test-time augmentation (TTA):** at the exam, show the model each photo and its mirror image and average the two answers.
- **AdamW + mixed precision:** a modern optimiser and a speed trick for the GPU.

### The ordinal head (our extra experiment)
Counting has an order: guessing 2 when the answer is 3 is a smaller mistake than guessing 5. A normal classifier treats all wrong answers as equally bad. The **ordinal head** instead answers five yes/no questions: "more than 0? more than 1? ... more than 4?". For 3 items the answers are yes, yes, yes, no, no. The count is the number of "yes" answers.

We trained both versions and compared them.

---

## 6. How we measure success

- **Accuracy:** how often the count is exactly right. Random guessing gets 16.7% (1 in 6).
- **RMSE:** the typical size of the error in items; lower is better. It punishes big mistakes more.
- **Within 1:** how often the count is off by at most one item.
- **Confusion matrix:** a table showing, for each true count, what the model predicted. The diagonal is correct answers.

---

## 7. Results (on the 3,000 test photos)

| Model | Accuracy | RMSE | Within 1 |
|---|---|---|---|
| Random guess | 16.7% | - | - |
| Always say "3" (majority class) | 25.3% | 1.35 | 68.8% |
| Logistic regression (PCA) | 28.5% | 1.37 | 73.1% |
| Linear SVM (PCA) | 27.7% | 1.37 | 73.0% |
| Decision tree (PCA) | 25.5% | 1.57 | 65.7% |
| RBF-SVM (PCA) | 29.8% | 1.32 | 74.9% |
| RBF-SVM (HOG) - best old-school | 31.9% | 1.32 | 75.3% |
| **ResNet34, softmax head** - best | **41.2%** | 1.11 | 83.4% |
| ResNet34, ordinal head | 39.4% | **1.04** | **86.6%** |

**What this tells us**
- The deep model beats the best old-school model by 29% (relative), and random guessing by 2.5 times.
- The ordinal head makes smaller mistakes (lower RMSE, more answers within 1) but is exact less often. So it is better when "close" matters, the normal head when "exact" matters.
- The model is best on empty bins (97.6% correct) and weakest on crowded bins (4 items: 24.5%), where items overlap and hide each other.

---

## 8. From counting to "reconciliation"

The public dataset only has true counts, so we **simulated** the mistakes in the inventory system: in 15% of test bins, the "recorded" number was changed by 1 or 2 items. Then the rule is: **flag the bin if the model's count differs from the recorded count.**

| Measure | Result | Meaning |
|---|---|---|
| Recall | 81.4% | Of the bins that truly had a mismatch, we caught 81% |
| Precision | 19.4% | Of the bins we flagged, only 19% truly had a mismatch |
| F1 | 0.31 | Combined score |
| AUROC | 0.73 | How well the model's confidence separates good bins from bad ones (0.5 = useless, 1 = perfect) |

**Honest reading:** the detector catches most real problems but raises many false alarms (it flags about 62% of bins), because the counting model is only 41% accurate. It is a promising proof of concept, not something Amazon could deploy yet. The mismatch errors are also simulated, not real.

---

## 9. Why not as good as the Stanford paper?

The paper we studied reports 56.2% accuracy; we got 41.2%. Likely reasons:
- The LR finder picked very small learning rates (about 0.00003 for the new layer, and the older layers moved even less), so the model was probably **under-trained**. This is our best guess, not proven.
- We used 30,000 photos; the paper's best run used 324,000.
- Different random sample and settings.

**The obvious fix:** re-train with a fixed learning rate of about 0.001 and no LR finder (around 30 minutes in Colab). We did not do this.

---

## 10. Likely questions in the review, with short answers

**Why use transfer learning?** It starts from a model that already understands images, so we need less data and less time.

**Why does the CNN beat SVM?** CNNs learn shapes and textures directly from pixels; PCA/HOG with an SVM only sees a compressed summary and loses detail.

**Why three different learning rates?** Early layers hold general knowledge that should barely change; the new layer starts random and must learn quickly.

**What is the validation set for?** Choosing settings (best model, best SVM parameters) without peeking at the test set, so the final score is honest.

**What does test-time augmentation do?** Averages predictions on the photo and its mirror, which reduces random mistakes (+0.5 accuracy points for us).

**Why is accuracy only 41%?** Occluded photos, similar-looking classes (3 vs 4 items), a smaller dataset than the paper, and likely under-training from a tiny learning rate.

**Is the 15% mismatch real?** No, it is simulated, because the public data has no real inventory records.

**Did the ordinal head help?** Partly: lower RMSE (1.04 vs 1.11), but lower accuracy (39.4% vs 41.2%).

**What would you do next?** Re-train with a higher fixed learning rate, use all ~324K images, try ensembles and higher resolution, and remove unreadable photos.

---

## 11. Quick glossary

- **Epoch:** one full pass through all the training photos.
- **Learning rate:** how big a step the model takes when it corrects itself.
- **Overfitting:** memorising the training photos instead of learning the idea.
- **Label:** the correct answer for a photo.
- **Softmax:** turns the model's scores into probabilities that add up to 100%.
- **GPU:** the fast chip (Colab's T4) that makes training quick.
