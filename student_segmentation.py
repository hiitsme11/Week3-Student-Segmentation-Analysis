from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
def main():
    df=pd.read_csv(ROOT/"data"/"student_activity.csv")
    features=["study_hours_per_week","attendance_percent","assignment_score","previous_exam_score","extracurricular_hours","class_participation"]
    X=df[features]
    prep=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    Xs=prep.fit_transform(X)
    rows=[]
    for k in range(2,7):
        labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xs)
        rows.append({"n_clusters":k,"silhouette_score":silhouette_score(Xs,labels),"davies_bouldin_score":davies_bouldin_score(Xs,labels)})
    selection=pd.DataFrame(rows); selection.to_csv(OUT/"kmeans_cluster_selection.csv",index=False)
    best_k=int(selection.loc[selection.silhouette_score.idxmax(),"n_clusters"])
    models={"KMeans":KMeans(n_clusters=best_k,random_state=42,n_init=10),"Agglomerative":AgglomerativeClustering(n_clusters=best_k,linkage="ward")}
    assigned=df.copy(); metric_rows=[]
    for name,model in models.items():
        labels=model.fit_predict(Xs); assigned[name.lower()+"_cluster"]=labels
        metric_rows.append({"model":name,"n_clusters":best_k,"silhouette_score":silhouette_score(Xs,labels),"davies_bouldin_score":davies_bouldin_score(Xs,labels),"calinski_harabasz_score":calinski_harabasz_score(Xs,labels)})
    metrics=pd.DataFrame(metric_rows); metrics.to_csv(OUT/"clustering_metrics.csv",index=False)
    assigned.to_csv(OUT/"student_segments.csv",index=False)
    coords=PCA(n_components=2,random_state=42).fit_transform(Xs)
    pca=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"cluster":assigned["kmeans_cluster"]})
    pca.to_csv(OUT/"pca_coordinates.csv",index=False)
    plt.figure(figsize=(8,6))
    for c in sorted(pca.cluster.unique()):
        q=pca[pca.cluster==c]; plt.scatter(q.PC1,q.PC2,label=f"Cluster {c}",alpha=.7)
    plt.title("Student Segments using K-Means (PCA view)"); plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2"); plt.legend(); plt.tight_layout(); plt.savefig(OUT/"student_segments_pca.png",dpi=150); plt.close()
    assigned.groupby("kmeans_cluster")[features].mean().round(2).to_csv(OUT/"cluster_profiles.csv")
    Z=linkage(Xs[:min(80,len(Xs))],method="ward")
    plt.figure(figsize=(12,6)); dendrogram(Z,truncate_mode="lastp",p=25,leaf_rotation=45,leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram (sample up to 80 students)"); plt.xlabel("Student groups"); plt.ylabel("Ward distance"); plt.tight_layout(); plt.savefig(OUT/"hierarchical_dendrogram.png",dpi=150); plt.close()
    (OUT/"run_summary.json").write_text(json.dumps({"rows":len(df),"features_used":features,"selected_cluster_count":best_k,"note":"Cluster numbers are arbitrary labels, not student ranks."},indent=2))
    print("Student Segmentation Analysis completed."); print("Rows analysed:",len(df)); print("Selected clusters:",best_k); print(metrics.to_string(index=False)); print("Outputs saved in:",OUT)
if __name__=="__main__": main()
