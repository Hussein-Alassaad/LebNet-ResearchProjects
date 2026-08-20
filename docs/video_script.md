# 3-Minute Video Script

Simple one-take script. Show your screen (GitHub repo / code / results files) while reading each part.

---

**[0:00–0:20] Intro**

"Hi, I'm Hussein. For this project I reproduced the paper 'XGBoost: A Scalable Tree Boosting System' by Chen and Guestrin, and then extended it by comparing XGBoost against two newer libraries — LightGBM and CatBoost."

*(Show: GitHub repo homepage)*

---

**[0:20–0:50] The paper & goal**

"XGBoost is a method for making predictions from tables of data using many small decision trees combined together. My goal was to rebuild the paper's method myself on a real dataset — the UCI Adult Census Income dataset — and check I could reproduce results in the range the community typically reports for this task."

*(Show: docs/paper_summary.md)*

---

**[0:50–1:30] What I built**

"I built the full pipeline from scratch: data loading and cleaning, the XGBoost model itself, a training script with early stopping, and evaluation code. Along the way I found a real bug — 52 duplicate rows in the dataset were leaking between my training and test sets, which was inflating my accuracy. I fixed that by deduplicating before splitting the data."

*(Show: src/ folder, then results/reproduction_results.md)*

---

**[1:30–2:00] Reproduction result**

"After fixing that and tuning hyperparameters, my final model reached 87.53% accuracy and 0.93 AUC on the test set — right in line with published reference numbers for this benchmark. I also verified this was truly reproducible by cloning the project fresh and getting the exact same result."

*(Show: results/fresh_clone_verification.md or the metrics table)*

---

**[2:00–2:40] The extension**

"For my extension, I asked: the paper claims XGBoost is both accurate and fast — how does it actually compare to the two libraries built specifically to beat it? I benchmarked XGBoost, LightGBM, and CatBoost with matched settings. All three ended up almost identically accurate — within 0.2 percent of each other. But the speed was very different: LightGBM trained the fastest, CatBoost was over 10 times slower to train but fastest at making predictions, and XGBoost landed right in the middle on both."

*(Show: results/extension_results.md or the comparison table)*

---

**[2:40–3:00] Wrap-up**

"So the takeaway is: modern gradient boosting libraries are all roughly equally accurate now — the real differences are in speed and engineering trade-offs, which is exactly what the original paper argued mattered. All my code, results, and write-up are on GitHub. Thanks for watching."

*(Show: GitHub repo one more time)*

---

## Recording tips

- Read at a natural pace — this script is timed for ~3 minutes total.
- Use free screen recording: **Windows Game Bar** (`Win+G`) or **OBS Studio**.
- Do one practice read-through before recording for real.
- If you stumble, just pause and keep going — you can trim silence later, or just re-record that section.
