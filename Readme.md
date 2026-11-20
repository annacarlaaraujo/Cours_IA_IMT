# TD – AI for Machining: Material Recognition from Spindle Power

Anna Carla Araujo – INSA Toulouse / Institut Clément Ader · November 2026

## Context and objective

You will build a model that recognizes the drilled material (aluminium, steel, titanium, or no material) using only the spindle power measured by the CNC. The same drill and the same cutting conditions were used in every test; only the workpiece material changes.

This TD applies what you practised in the Machine Learning labs at IMT Mines Albi to real machining data. 

Before any model, you will use machining knowledge to understand the signal: drilling phases, spindle speed, torque, and cutting force.

**Work in Google Colab, in groups of 3.** 

## Experimental setup and data

The tests were run at INSA Toulouse on a CNC machining centre with a Siemens SINUMERIK controller. 
The controller trace recorded the spindle active power every 2 ms (500 Hz).

| Parameter | Value |
| --- | --- |
| Tool | Twist drill, 2 cutting lips, diameter D = 11 mm, Inner diameter = 4mm, Point angle 120 degrees |
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



## Part A — Understanding the process and the signal 

No machine learning yet: the goal is to know what the curves mean before asking a model to use them.

1 - What is the cutting speed variation along the cutting edge (m/min) ?

2 - What is the feed velocity in z direction (mm/min) ?

3 - What is the time of the cutting, in minutes and in seconds ?

4 - What is the area of cutting, after the tool point is inside the hole  (mm^2) ?

5 -  How is the variation of the cutting force of one cutting edge and the power of two cutting edges, if we consider Kc=700 MPa ?


6 - Plot one curve (any curve) and look the format of the curve. There is a plateau ? How many seconds of cutting ? Could you plot in terms of z position (z=0 is the surface of the piece) ?

7 - Is there a plateau before cutting ? This is the idle power. 

8 -  Calculate the power plateau related to the cutting power : Pc(only cutting) = P(measured) − P(idle)

9 -  What would it be the Kc (MPa) of this curve ?


## Part B — Features and models 

Think about how to construct a Python code to identify the curve and to classify the curve.

You now can turn each cycle into a small set of numbers (features) and train classifiers, using the same tools as in the IMT Mines Albi labs (Prof. Jad).

a) Write a function `features(df)` that returns at least 3 physically justified features for one cycle. Examples: idle power, maximum smoothed power, Pc of each pass, duration of the cutting phases, kc from A.9. Build `X_train` (84 rows) and `y_train` (4 classes). Could you justify each feature ?

b) Split the 84 labelled cycles into training (75 %) and validation (25 %) with `train_test_split(..., stratify=y, random_state=0)`. Fit `StandardScaler` on the training part only. Why must the scaler not see the validation data?

c) Train a `LogisticRegression` baseline (as your previous class : https://github.com/JadM133/Cours_IA_IMT/blob/main/labs/01_MLP_Binary_Classification_solution.ipynb . 

What is the accuracy and plot the confusion matrix (`ConfusionMatrixDisplay`). 

## Part C - Extras (if you want to go deeper) :

Could you build a PyTorch MLP with `nn.Sequential`: n_features → 16 → ReLU → 16 → ReLU → 4 logits.  ?

you can use `nn.CrossEntropyLoss()` and Adam (learning rate 0.01) and train with the five familiar steps (clear gradients, forward pass, loss, backward pass, update). Compare its validation accuracy with the baseline. Is the neural network useful here? Explain.

*Hint:* this is a 4-class problem, so the labels are integers 0–3 (`torch.long`) and the prediction is `logits.argmax(dim=1)`, not a sigmoid threshold.

Choose your final model using the validation results only. Retrain it on all 84 labelled cycles, then predict the class of the 36 files in `test/`. Save the predictions as `predictions.csv` with two columns, `file;predicted_class` (for example `hole_01;Steel`). Send this file to the instructor by email: you will receive your test accuracy. The true classes of the test files are not released.

Which single feature separates the materials best, and what does it mean physically?

## What to keep for the exam

The course exam will include a question on this TD. Keep your notebook and save your figures as you go.

Figures worth keeping:

- One power cycle per class with the drilling phases marked 
- Table of kc per material 
- Learning curve: validation accuracy against the number of training cycles per class
Extra:
- Confusion matrix of your logistic regression on the validation set, and your test accuracy 
