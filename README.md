# Student Segmentation Analysis

This project groups **600 synthetic/demo student records** using K-Means and Agglomerative (Hierarchical) Clustering. It demonstrates preprocessing, feature scaling, cluster evaluation, PCA visualization and cluster profiling. No real student personal information is included.

Features: study hours per week, attendance percentage, assignment score, previous exam score, extracurricular hours and class participation. Student ID is not used as a model feature.

The script imputes missing values with the median, standardizes the features, tests K-Means cluster counts from 2 to 6 using Silhouette Score, then compares K-Means and Agglomerative clustering using Silhouette, Davies-Bouldin and Calinski-Harabasz scores. PCA is used only for a two-dimensional plot. A dendrogram and average cluster profiles are also generated.

## Windows PowerShell setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python student_segmentation.py
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Outputs are saved in `outputs/`: clustering metrics, cluster selection scores, segment assignments, cluster profiles, PCA coordinates, PCA scatter plot, dendrogram and run summary.

Cluster IDs are arbitrary labels, not student ranks or grades. Use this synthetic dataset for practice; do not make high-stakes decisions about students from unsupervised clusters.
