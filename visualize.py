import plotly.express as px
from sklearn.datasets import load_iris
from umap import UMAP

def main():
    # Load the Iris data
    iris = load_iris()
    umap_2d = UMAP()
    # Fit UMAP to Iris data
    umap_2d.fit(iris.data)
    projections = umap_2d.transform(iris.data)
    
    # Plot the projections with the Iris target as color
    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=iris.target.astype(str),
        labels={"color": "iris"}
    )
    
    # Save the plot as an HTML file
    fig.write_html("public/index.html")

