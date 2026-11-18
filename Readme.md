# TD – AI for Machining: Material Recognition from Spindle Power

Anna Carla Araujo – INSA Toulouse / Institut Clément Ader · November 2026

## Context and objective

You will build a model that recognizes the drilled material (aluminium, steel, titanium, or no material) using only the spindle power measured by the CNC. The same drill and the same cutting conditions were used in every test; only the workpiece material changes.

This TD applies what you practised in the Machine Learning labs at IMT Mines Albi (Institut Mines-Télécom) (train/test split, scaling without leakage, logistic regression baseline, PyTorch MLP, confusion matrix) to real machining data. Before any model, you will use machining knowledge to understand the signal: drilling phases, spindle speed, torque, and cutting force.

**Work in Google Colab, in pairs.** Part A and questions B1–B3 are done during the TD session (1h30). Questions B4–B5 and Part C are completed at home. The steps take you from the raw signal to the results you will present in the course exam (see the last section). No report is graded.

## Experimental setup and data

The tests were run at INSA Toulouse on a CNC machining centre with a Siemens SINUMERIK controller (16 April 2025). The controller trace recorded the spindle active power every 2 ms (500 Hz).

| Parameter | Value |
| --- | --- |
| Tool | Twist drill, 2 cutting lips, diameter D = 11 mm |
| Spindle speed | N = 1085 rev/min |
| Feed | f = 0.1 mm/rev |
| Cycle | Peck drilling: 2 passes of 6 mm, retract between passes |
| Hole depth | 12 mm |
| Materials | Aluminium alloy, steel, titanium alloy, and air (cycle run without a workpiece) |
| Signal | Spindle active power P (W), sampling period 2 ms |

Each file contains one complete drilling cycle (approach, two passes, retract) with two columns: `t_s` (time from the start of the cycle, s) and `power_W` (spindle active power, W).

| Folder | Content | Files |
| --- | --- | --- |
| `train/` | Labelled cycles: the file name gives the class | `Aluminium_01…25`, `Steel_01…31`, `Titanium_01…14`, `Air_01…14` (84 files) |
| `test/` | Unlabelled cycles to classify | `hole_01…36` (36 files) |



## Part A — Understanding the signal (in class)

No machine learning yet: the goal is to know what the curves mean before asking a model to use them.

**A1.** Plot one cycle of each class on the same figure. The raw signal is noisy; also plot a moving average over 0.1 s (50 samples). Describe in two sentences how the four classes differ.

**A2.** On one steel cycle, mark the phases of the drilling cycle: approach (tool in air), entry of the drill point, first pass, retract and chip evacuation, second pass, final retract. Which signal feature tells you that the drill is cutting? Why does the power rise gradually at the start of the first pass?

**A3.** Compute the amplitude spectrum (FFT) of the power during an air-cutting phase and identify the main frequency. Show that it matches the passage of the 2 cutting lips at N = 1085 rev/min (2 × N / 60). Then compute the feed velocity vf = f × N (mm/min).

*Hint:* `np.fft.rfft` and `np.fft.rfftfreq(n, d=0.002)`; remove the mean before the FFT.

**A4.** For each material, estimate the cutting power of the second pass, Pc = P(cutting) − P(idle), where P(idle) is the power just before the drill enters the material. Then compute:

```math
M_c = \frac{P_c}{\omega}, \qquad \omega = \frac{2\pi N}{60}
```

```math
F_c \approx \frac{2\,M_c}{D} \quad \text{(cutting force per lip, applied at a radius } D/4\text{)}
```

```math
k_c = \frac{P_c}{Q}, \qquad Q = \frac{\pi D^2}{4}\, v_f \quad \text{(specific cutting energy, N/mm}^2\text{)}
```

Present Mc, Fc and kc for the three materials in a table. Compare kc with typical values from your machining course. Why is your kc likely higher than the handbook value? Which force of the drilling process can you **not** obtain from the spindle power?

## Part B — Features and models (B1–B3 in class, B4–B5 at home)

You now turn each cycle into a small set of numbers (features) and train classifiers, using the same tools as in the IMT Mines Albi labs.

**B1.** Write a function `features(df)` that returns at least 3 physically justified features for one cycle. Examples: idle power, maximum smoothed power, Pc of each pass, duration of the cutting phases, kc from A4. Build `X_train` (84 rows) and `y_train` (4 classes). Justify each feature in one sentence.

**B2.** Split the 84 labelled cycles into training (75 %) and validation (25 %) with `train_test_split(..., stratify=y, random_state=0)`. Fit `StandardScaler` on the training part only. Why must the scaler not see the validation data?

**B3.** Train a `LogisticRegression` baseline. Report the validation accuracy and plot the confusion matrix (`ConfusionMatrixDisplay`). Which classes, if any, are confused?

*End of the TD session. Questions B4 and B5 are completed at home.*

**B4.** Build a PyTorch MLP with `nn.Sequential`: n_features → 16 → ReLU → 16 → ReLU → 4 logits. Use `nn.CrossEntropyLoss()` and Adam (learning rate 0.01). Train with the five familiar steps (clear gradients, forward pass, loss, backward pass, update). Compare its validation accuracy with the baseline. Is the neural network useful here? Explain.

*Hint:* this is a 4-class problem, so the labels are integers 0–3 (`torch.long`) and the prediction is `logits.argmax(dim=1)`, not a sigmoid threshold.

**B5.** How many training cycles do you need? Train the baseline with 1, 2, 3, 5 and 8 cycles per class (random draws, repeated 20 times) and plot the mean validation accuracy against the number of cycles. Where does the curve reach a plateau?

## Part C — Test set and conclusions (at home)

**C1.** Choose your final model using the validation results only. Retrain it on all 84 labelled cycles, then predict the class of the 36 files in `test/`. Save the predictions as `predictions.csv` with two columns, `file;predicted_class` (for example `hole_01;Steel`). Send this file to the instructor by email: you will receive your test accuracy. The true classes of the test files are not released.

**C2.** Answer in a few sentences each:

1. Which single feature separates the materials best, and what does it mean physically?
2. In an earlier version of this TD, students received the machine Z position. The workpieces of each material had different heights. Why would a model reach 100 % accuracy with that signal, and why would it be useless in production?
3. All aluminium holes were drilled first, then steel, then titanium, with the same drill. Name one factor other than the material that could also change the power. How would you design the experiment to rule it out?
4. Give one industrial use of this model and one limitation.

## What to keep for the exam

The course exam will include a question on this TD: you will explain how you proceeded, the difficulties you met, and present one or two figures of your results. Keep your notebook and save your figures as you go.

Figures worth keeping:

- One power cycle per class with the drilling phases marked (A1–A2)
- Table of Pc, Mc, Fc and kc per material (A4)
- Learning curve: validation accuracy against the number of training cycles per class (B5)
- Confusion matrix of your logistic regression on the validation set (B3), and your test accuracy (C1)

**Optional extensions, if you finish early:**

- **Without labels.** Apply `PCA(n_components=2)` and `KMeans(n_clusters=4)` to the scaled training features, ignoring the labels. Compare the clusters with the true classes using `adjusted_rand_score`. Would the materials be found without labels?
- **Raw curves.** Instead of hand-made features, resample each cycle to 450 points (every 10th sample) and train an MLP or a 1D CNN (`nn.Conv1d`) directly on the curve. Does it beat your feature-based model?
- **Unknown class.** Retrain without the `Air` cycles and predict the air cycles of the test set. What does the model answer? How could you detect a cycle that belongs to none of the trained classes?
