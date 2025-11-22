# DVC Setup and Data Management Guide

## Overview

This project uses DVC (Data Version Control) to manage large dataset files in a reproducible and versioned way. The MovieLens dataset contains files that exceed GitHub’s 100 MB limit, which means the raw data cannot be committed to the repository.  
DVC allows us to track the version of the dataset without storing the actual file in Git, while keeping the repository efficient and easy to clone.

This document explains why DVC is used, how to obtain the dataset, and what steps team members must follow to keep their local environment aligned with the project.

---

## Why We Use DVC

The file `ratings.dat` in the `ml-10M100K` directory is approximately 250 MB. This creates several challenges.

### GitHub Limitations

GitHub blocks any push that contains files larger than 100 MB.

### Git Limitations

Git does not handle large binary files efficiently:

- repository size increases quickly  
- history becomes unnecessarily large  
- cloning becomes slow  
- diffs and merges are ineffective  

### Reproducibility Requirements

All team members need to use the same version of the dataset for training and evaluation.

### How DVC Solves This

DVC keeps large files **outside Git**, while still allowing us to:

- track dataset versions through small `.dvc` pointer files  
- keep the Git repository small  
- reproduce experiments reliably  
- share metadata instead of raw data  

The actual dataset stays local or in DVC remote storage. Git only tracks the lightweight pointer file:

```
ml-10M100K/ratings.dat.dvc
```

## Repository Structure After DVC Integration

The project now contains:

```
.dvc/ DVC metadata (tracked)
ml-10M100K/
ratings.dat.dvc - Pointer file tracked by Git
ratings.dat - Actual dataset (ignored by Git)
movies.dat - Supporting file (tracked)
tags.dat - Supporting file (tracked)
allbut.pl - Script (tracked)
split_ratings.sh - Script (tracked)
README.html - Documentation (tracked)
.gitignore - Updated to exclude data and DVC cache
```

The `.gitignore` ensures that Git never tracks:

- `ml-10M100K/ratings.dat`
- `.dvc/cache/`
- `.dvc/tmp/`

## Required Steps for All Team Members

If you have already cloned the repository and are working in the `dev` branch, follow the steps below.

### 1. Pull the Latest Changes

```bash
git checkout dev
git pull origin dev
```

## Required Steps for All Team Members

If you have already cloned the repository and are working in the `dev` branch, follow the steps below.

### 1. Pull the Latest Changes

```bash
git checkout dev
git pull origin dev
```

### 2. Install DVC with Azure support(One-Time Step)
Install using pip 

```
pip install "dvc[azure]"
```

Verify installation:
```
dvc --version
```

### 3. Configure Azure Credentials
You must export two environment variables locally. These allow DVC to authenticate with Azure.
Replace values with the team-provided credentials.

```
export AZURE_STORAGE_ACCOUNT="your-storage-account-name"
export AZURE_STORAGE_KEY="your-access-key"

```

These must not be commited to Git or stored in the repository. 

### 4. Pull the Dataset from Azure

Once credentials are set, run:

```
dvc pull
```

DVC will download the dataset from Azure Blob Storage into:

```
ml-10M100K/ratings.dat
```

Important: do not commit `ratings.dat` to Git. It must remain ingored.

## Ongoing Workflow

Once setup is complete, you can continue using Git normally. You commit code, configuration, and `.dvc` pointer files, but never the raw data.
If the dataset version changes in the future, a new `.dvc` pointer file will be committed. Other teammates can sync their dataset by running:

```
git pull
dvc pull

```

This keeps project code and dataset versions consistent across the team.