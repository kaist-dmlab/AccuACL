import io
import base64
import pickle
import gzip
import numpy as np

# from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots
# from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go

from PIL import Image

from sklearn.manifold import TSNE

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from sklearn.metrics.pairwise import cosine_similarity

def get_model(name):
        if name == 'SplitCIFAR10':
            from avalanche.models.resnet18_32 import ResNet18_32, BasicBlock
            return ResNet18_32(BasicBlock, [2, 2, 2, 2], norm=True)
        else:
            print('no matching scenario')
            return

def get_forward(model, train_loader_past, train_loader_current, device):
    model.eval()
    imgs = []
    embs = []
    labels = []
    unlabeled = []
    logits=[]
    with torch.no_grad():
        for features,targets,idxs in tqdm(train_loader_past):
            labels.append(targets)
            imgs.append(features)
            logit, emb = model(features.to(device), repr=True)
            embs.append(emb.detach().cpu().numpy())
            probs = F.softmax(logit.detach().cpu(), dim=1).numpy()
            logits.append(probs)
            unlabeled+=[0]*len(targets)

        for features,targets,idxs in tqdm(train_loader_current):
            labels.append(targets)
            imgs.append(features)
            logit, emb = model(features.to(device), repr=True)
            embs.append(emb.detach().cpu().numpy())
            probs = F.softmax(logit.detach().cpu(), dim=1).numpy()
            logits.append(probs)
            unlabeled+=[1]*len(targets)
            
    imgs = np.vstack(imgs)
    embs = np.vstack(embs)
    logits = np.vstack(logits)
    labels = np.concatenate(labels).ravel()
    return imgs, embs, labels, unlabeled, logits

def deepal_forward(model, memory, current, device, temp=1.0):
    imgs = []
    embs = []
    labels = []
    unlabeled = []
    logits =[]

    imgs.extend([data[0] for data in memory])
    imgs.extend([data[0] for data in current])
    embs.append(model.get_embeddings(memory))
    embs.append(model.get_embeddings(current))
    labels.extend([data[1] for data in memory])
    labels.extend([data[1] for data in current])
    unlabeled+=[0]*len(memory)
    unlabeled+=[1]*len(current)
    logits.append(model.predict_logit(memory))
    logits.append(model.predict_logit(current))
            
    embs = np.vstack(embs)
    logits = np.vstack(logits)
    return imgs, embs, labels, unlabeled, logits

import matplotlib.patches as mpatches
import pandas as pd
def get_df(imgs, embs, labels, unlabeled, logits, res_pca, res_tsne):
    probability = F.softmax(torch.tensor(logits), dim=1)
    df_data = pd.DataFrame(data=res_pca, columns=['pca1', 'pca2'])
    df_data["target"] = labels
    df_data['current'] = unlabeled
    df_data['id'] = df_data.index
    df_data['imgs'] = [np.array(i) for i in imgs]
    df_data['embs'] = [i for i in embs]
    df_data['probability'] = [i for i in probability]
    df_data['pred'] = df_data['probability'].apply(lambda x: x.argmax())
    df_data['inference'] = df_data['pred']==df_data['target']
    df_data['tsne1'] = res_tsne[:,0]
    df_data['tsne2'] = res_tsne[:,1]
    df_data['alpha'] = df_data['inference'].apply(lambda x: 0.11 if x else 1)
    ent = Categorical(probs=torch.tensor(probability)).entropy().detach().cpu().numpy()/np.log(10)
    df_data['entropy'] = [i for i in ent]
    df_mem = df_data[df_data['current']==0].copy().reset_index()
    # gmm = generate_gaussian_prototypes(df_mem.target.unique(), df_mem)
    # proto_preds = proto_dist_classification(df_data['embs'].tolist(), gmm)
    # df_data['proto_pred'] = proto_preds

    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#0080FF', '#8000FF', '#FF0080']
    df_data['color'] = df_data['target'].apply(lambda x: colors[x])
    df_data['pred_color'] = df_data['pred'].apply(lambda x: colors[x])
    # df_data['proto_pred_color'] = df_data['proto_pred'].apply(lambda x: colors[x])
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, range(len(colors)))]
    df_current = df_data[df_data['current']==1].copy().reset_index()
    df_mem = df_data[df_data['current']==0].copy().reset_index()
    # df_current['novelty'] = min_max_norm(get_novelty_min_dist(df_current['embs'].tolist(), gmm))
    # df_current['expressibility'] = 1-df_current['novelty']
    # df_current['mahalanobis_novelty'] = mahalanobis_novelty(df_current['embs'].tolist(), gmm)
    return df_data, df_current, df_mem, legend_patches

def min_max_norm(data):
    return (data-data.min())/(data.max()-data.min())

def get_fim_df(tsne, fims, fim, labels):
    df_data = pd.DataFrame(data=tsne, columns=['1','2'])
    df_data['target'] = labels
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000', '#0080FF', '#8000FF', '#FF0080']
    df_data['color'] = df_data['target'].apply(lambda x: colors[x])
    df_data['align'] = min_max_norm(cosine_similarity(fims, fim.unsqueeze(0)))
    df_data['norm'] = min_max_norm(fims.norm(dim=1))
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, range(len(colors)))]
    return df_data, legend_patches


def get_torchvision_embs(model, features):
    return torch.nn.Sequential(*list(model.children())[:-1])(features).squeeze()

def get_torchvision_forward(model, train_loader_past, train_loader_current, device, temp=1.0):
    imgs = []
    embs = []
    labels = []
    unlabeled = []
    uncertainty = []
    probability =[]
    model.eval()
    with torch.no_grad():
        for i, (features, targets) in enumerate(tqdm(train_loader_past)):
            embs.append(get_torchvision_embs(model, features.to(device)).detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            imgs.append(features)
            logits = get_torchvision_embs(model, features.to(device))
            probs = F.softmax(logits/temp, dim=1)
            probability.append(probs.detach().cpu().numpy())
            probs_sorted = torch.sort(probs, 1, descending=True).values.detach().cpu().numpy()
            ent = Categorical(probs=probs).entropy().detach().cpu().numpy()
            uncertainty.append(ent)
            unlabeled+=[0]*len(targets)

        for i, (features, targets) in enumerate(tqdm(train_loader_current)):
            embs.append(get_torchvision_embs(model, features.to(device)).detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            imgs.append(features)
            logits = get_torchvision_embs(model, features.to(device))
            probs = F.softmax(logits/temp, dim=1)
            probability.append(probs.detach().cpu().numpy())
            probs_sorted = torch.sort(probs, 1, descending=True).values.detach().cpu().numpy()
            ent = Categorical(probs=probs).entropy().detach().cpu().numpy()
            uncertainty.append(ent)
            unlabeled+=[1]*len(targets)
            
    imgs = np.vstack(imgs)
    embs = np.vstack(embs)
    uncertainty = np.concatenate(uncertainty).ravel()
    labels = np.concatenate(labels).ravel()
    probability = np.vstack(probability)
    return imgs, embs, labels, unlabeled, uncertainty, probability


cmap = ['#636EFA', # the plotly blue you can see above
 '#EF553B',
 '#00CC96',
 '#AB63FA',
 '#FFA15A',
 '#19D3F3',
 '#FF6692',
 '#B6E880',
 '#FF97FF',
 '#FECB52']

def np_image_to_base64(im_matrix, rgb, mean, std):
    im_matrix = im_matrix*std+mean
    im_matrix = (im_matrix*255).astype(np.uint8)
    if rgb:
        im_matrix = im_matrix.transpose((1,2,0))
    else:
        im_matrix = im_matrix[0]
    im = Image.fromarray(im_matrix)
    if rgb:
        im = im.convert("RGB")
    else:
        im = im.convert("L")
    buffer = io.BytesIO()
    im.save(buffer, format="png")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/png;base64, " + encoded_image
    return im_url

class plotly_plot():
    def __init__(self, class_names, curr, mem, is_rgb=False, mean=None, std=None, color='target', tsne=True):
        self.class_names=class_names
        self.curr=curr
        self.mem=mem
        self.is_rgb = is_rgb
        self.std = std
        self.mean = mean
        if self.is_rgb:
            self.std = np.broadcast_to(np.array(std).reshape(3,1,1), self.curr.imgs[0].shape)
            self.mean = np.broadcast_to(np.array(mean).reshape(3,1,1), self.curr.imgs[0].shape)
        
    def draw_plot(self, port=8050):
        curr_X = self.curr['tsne1']
        curr_Y = self.curr['tsne2']
        mem_X = self.mem['tsne1']
        mem_Y = self.mem['tsne2']
        
        fig = go.Figure(data = [
            go.Scatter(x=curr_X, y=curr_Y, mode='markers', marker=dict(size=10, color=self.curr.novelty.tolist())),
            go.Scatter(x=mem_X, y=mem_Y, mode='markers', marker=dict(size=10, color=self.mem.color.to_list()), showlegend=True),
            ]
        )
        
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        fig.update_layout(
            autosize=False,
            width=1600,
            height=1600,
        )
        app = JupyterDash(__name__)
        app.layout = html.Div(
            className="container",
            children=[
                dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
            ],
        )
        
        @app.callback(
            Output("graph-tooltip-5", "show"),
            Output("graph-tooltip-5", "bbox"),
            Output("graph-tooltip-5", "children"),
            Input("graph-5", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update
            # demo only shows the first point, but other points may also be available
            
            hover_data = hoverData["points"][0]
            mem = hover_data['curveNumber']
            bbox = hover_data["bbox"]
            num = hover_data["pointNumber"]
            
            if mem:
                data = self.mem
            else:
                data = self.curr
                
            im_matrix = data.imgs[num]
            im_url = np_image_to_base64(im_matrix, self.is_rgb, self.mean, self.std)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
                    ),
                    html.P(str(self.class_names[data.target[num]])+': '+str(data.target[num]), style={'font-weight': 'bold'}),
                    html.P('unlabeled: '+str(data['current'][num])),
                    html.P('index: '+str(num) )
                ])
            ]

            return True, bbox, children
        app.run_server(host='dmserver1.kaist.ac.kr', port=port, debug=True)

class plotly_scatter_plot():
    def __init__(self, class_names, data, is_rgb=False, mean=None, std=None, color='target', tsne=True):
        self.class_names = class_names
        self.data = data
        self.labels = self.data['target']
        self.unlabeled = self.data['current']
        self.data['id'] = self.data.index
        self.is_rgb = is_rgb
        self.data['class'] = self.labels.apply(lambda x: class_names[x])
        self.std = std
        self.mean = mean
        self.color = color
        self.tsne = tsne
        if self.is_rgb:
            self.std = np.broadcast_to(np.array(std).reshape(3,1,1), self.data.imgs[0].shape)
            self.mean = np.broadcast_to(np.array(mean).reshape(3,1,1), self.data.imgs[0].shape)
    
    
    def draw_plot(self, port=8050):
        if self.tsne:
            X=self.data['tsne1']
            Y=self.data['tsne2']
        else:
            X=self.data['pca1']
            Y=self.data['pca2']
        fig = go.Figure(data=[go.Scatter(
            x=X,
            y=Y,
            mode='markers',
            marker=dict(
                size=10,
                color=[cmap[c] for c in self.data[self.color].to_list()],
            )
        )])

        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        fig.update_layout(
            autosize=False,
            width=1600,
            height=1600,
        )
        app = JupyterDash(__name__)
        app.layout = html.Div(
            className="container",
            children=[
                dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
            ],
        )
        
        @app.callback(
            Output("graph-tooltip-5", "show"),
            Output("graph-tooltip-5", "bbox"),
            Output("graph-tooltip-5", "children"),
            Input("graph-5", "hoverData"),
        )
        def display_hover(hoverData):
            if hoverData is None:
                return False, no_update, no_update
            # demo only shows the first point, but other points may also be available
            hover_data = hoverData["points"][0]
            bbox = hover_data["bbox"]
            num = hover_data["pointNumber"]

            im_matrix = self.data.imgs[num]
            im_url = np_image_to_base64(im_matrix, self.is_rgb, self.mean, self.std)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
                    ),
                    html.P(str(self.class_names[self.data.target[num]])+': '+str(self.data.target[num]), style={'font-weight': 'bold'}),
                    html.P('unlabeled: '+str(self.data['current'][num])),
                    html.P('index: '+str(num) )
                ])
            ]

            return True, bbox, children
        app.run_server(host='dmserver5.kaist.ac.kr', port=port, debug=True)

import random
from types import SimpleNamespace
from typing import Dict, Union

import numpy as np
import torch

from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheSubset


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False


def create_default_args(args_dict, additional_args=None):
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        for k, v in additional_args.items():
            args.__dict__[k] = v
    return args


def restrict_dataset_size(scenario, size: int):
    """
    Util used to restrict the size of the datasets coming from a scenario
    param: size: size of the reduced training dataset
    """
    modified_train_ds = []
    modified_test_ds = []
    modified_valid_ds = []

    for i, train_ds in enumerate(scenario.train_stream):
        train_ds_idx, _ = torch.utils.data.random_split(
            torch.arange(len(train_ds.dataset)),
            (size, len(train_ds.dataset) - size),
        )
        dataset = AvalancheSubset(train_ds.dataset, train_ds_idx)

        modified_train_ds.append(dataset)
        modified_test_ds.append(scenario.test_stream[i].dataset)
        if hasattr(scenario, "valid_stream"):
            modified_valid_ds.append(scenario.valid_stream[i].dataset)

    scenario = dataset_benchmark(
        modified_train_ds,
        modified_test_ds,
        other_streams_datasets={"valid": modified_valid_ds}
        if len(modified_valid_ds) > 0
        else None,
    )

    return scenario