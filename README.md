# OKS

**Calculate OKS(Object Keypoints Similarity) from two sets of keypoints**

OKS is defined in [here](https://cocodataset.org/#keypoints-eval).

## Arguments

To caluculate OKS, you need four arguments; `kpts1`, `kpts2`, `sigma`, and `area`.

### `kpts1` & `kpts2`: Sets of Keypoints

The `OKS` function caluclates the value of OKS between `kpts1` and `kpts2`.

`kpts1` and `kpts2` must be same shape and each keypoint must have three parameters; x, y, and v (v is a visibility).

### `sigma`: Per-Keypoint Standard Deviation

`sigma` is a set of parameters of per-keypoint standard deviation. It is determined for each dataset. Detail is [here](https://cocodataset.org/#keypoints-eval).

Samples of `sigma`:
```python
# COCO
sigma = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

# body_25
sigma = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .8, .8, .8, .89, .89, .89, .89, .89, .89])/10.0
```

If you use your custom dataset, you have to prepare your `sigma`.

### `area`: Area of the Object

`area` is the number of pixels of the object (such as person) in the picture.

For sake of simplicity, you can use the mean of "area" in your COCO annotation.

## How to Use

You can caluclate the value of OKS between `kpts1` and `kpts2` like below:

```python
# This is the sample for COCO keypoints.
sigma = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0


```

## Calculate `sigma` for Your Custom Dataset

You can get your `sigma` from your custom dataset which is annotated redundantly, or caluclate pseudo `sigma` using the diffelence between ground truth and detected keypoints (code is coming soon!).