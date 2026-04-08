import kagglehub

path = kagglehub.dataset_download(
    "mbornoe/lisa-traffic-light-dataset", output_dir="../data/raw/"
)
print("Path to dataset files:", path)

path = kagglehub.dataset_download(
    "researcherno1/small-traffic-lights", output_dir="../data/raw/"
)
print("Path to dataset files:", path)
