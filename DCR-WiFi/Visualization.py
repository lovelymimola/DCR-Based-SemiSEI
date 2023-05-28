import torch
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from VAT_DataAugmentation_5label_rotate2 import *
import sklearn.metrics as sm
from sklearn import manifold

def scatter(features, targets, subtitle = None, n_classes = 16):
    palette = np.array(sns.color_palette("hls", n_classes))  # "hls",
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])  #
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.savefig(f"Visualization/{n_classes}classes_{subtitle}.png", dpi=600)

def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure()
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)

def obtain_embedding_feature_map(model, test_dataloader):
    model.eval()
    device = torch.device("cuda:0")
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            #target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                #target = target.to(device)
            output = model(data)
            feature_map[len(feature_map):len(output[0])-1] = output[0].tolist()
            target_output[len(target_output):len(target)-1] = target.tolist()
        feature_map = torch.Tensor(feature_map)
        target_output = np.array(target_output)
    return feature_map, target_output

def main():
    ft = 62
    X_test, Y_test = TestDataset(ft)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = torch.load('model_weight/CNN_FixMatch_SSML_Norm_Rotate2_n_classes_16_5label_95unlabel_rand30.pth')
    X_test_embedding_feature_map, target = obtain_embedding_feature_map(model, test_dataloader)

    tsne = TSNE(n_components=2)
    eval_tsne_embeds = tsne.fit_transform(torch.Tensor.cpu(X_test_embedding_feature_map))
    scatter(eval_tsne_embeds, target.astype('int64'), "CNN_MAT_n_classes_16_5label_95unlabel", 16)
    visualize_data(X_test_embedding_feature_map, target.astype('int64'), "Visualization/CNN_MAT_n_classes_16_5label_95unlabel_improved", 16)
    print(sm.silhouette_score(X_test_embedding_feature_map, target, sample_size=len(X_test_embedding_feature_map), metric='euclidean'))

if __name__ == "__main__":
    main()