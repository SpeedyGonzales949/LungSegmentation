from preporcess import create_groups,find_empy

in_dir="C:/Users/mihne/Desktop/Accenture-Collab/Datasets/COVID-19-20_v2/Diicom/TestVolumes"
out_dir="C:/Users/mihne/Desktop/Accenture-Collab/Datasets/COVID-19-20_v2/Data/TestVolumes"

# data_dir = 'C:/Users/mihne/Desktop/Accenture-Collab/LiverSeg-using-monai-and-pytorch/Liver-Segmentation-Using-Monai-and-PyTorch/LiverSegementationDataSet/TrainSegmentation'
# print(find_empy(data_dir))
create_groups(in_dir,out_dir,32)