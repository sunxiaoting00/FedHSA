import time
import numpy as np
from flcore.clients.clienttoyexample import clientProtoexample
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.metrics import pairwise_distances
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import torch
import seaborn as sns

class FedProtoexample(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientProtoexample)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.threshold = 10

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()
               # if i == 300:
                #    self.visualize_tsne_features(i)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos(i)
           # if i == 300:
            #    self.visualize_tsne_prototypes(i)
             #   self.visualize_tsne_prototypes2(i)

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            # t-SNE protos visualize

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def receive_protos(self, i):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        uploaded_protos = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            uploaded_protos.append(protos)

        global_classprotos, global_protos= self.new_proto_aggregation(uploaded_protos)
        save_item(global_classprotos, self.role, 'global_classprotos', self.save_folder_name)
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
    """
    def split_protos_by_similarity(self, uploaded_protos):
        
        class_protos = defaultdict(list)

        for client_protos in uploaded_protos:
            for label, proto_tensor in client_protos.items():
                for single_proto in proto_tensor:
                    class_protos[label].append(single_proto.detach().cpu())

        sub_class_protos = {}

        for label, proto_list in class_protos.items():
            proto_array = torch.stack(proto_list).numpy()  # shape: [N, dim]
            sim_matrix = cosine_similarity(proto_array)

            clusters = {}
            assigned = [-1] * len(proto_array)
            cluster_id = 0

            for i in range(len(proto_array)):
                if assigned[i] == -1:
                    clusters[cluster_id] = [i]
                    assigned[i] = cluster_id
                    for j in range(i + 1, len(proto_array)):
                        if assigned[j] == -1 and sim_matrix[i, j] >= self.tau_intra:
                            clusters[cluster_id].append(j)
                            assigned[j] = cluster_id
                    cluster_id += 1

            for sub_id, indices in clusters.items():
                cluster_proto_list = [proto_list[idx] for idx in indices]  # List[Tensor]

                if len(cluster_proto_list) > 1:
                    proto = 0 * cluster_proto_list[0].data
                    for p in cluster_proto_list:
                        proto += p.data
                    avg_proto = proto / len(cluster_proto_list)
                else:
                    avg_proto = cluster_proto_list[0].data

                sub_label = f"{label}_sub{sub_id + 1}"
                sub_class_protos[sub_label] = avg_proto.to(self.device)

        return sub_class_protos
"""

    def estimate_dynamic_threshold(self,distance_matrix, percentile=30):
        
        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
        upper_tri_values = distance_matrix[upper_tri_indices]

        finite_vals = upper_tri_values[np.isfinite(upper_tri_values)]
        if len(finite_vals) == 0:
            return np.inf  # fallback

        threshold = np.percentile(finite_vals, percentile)
        return threshold

    def new_proto_aggregation(self, local_protos_list):
        agg_protos = defaultdict(list)

        for local_protos in local_protos_list:
            for label, proto_list in local_protos.items():
                agg_protos[label].extend(proto_list)

        global_classprotos = {}
        global_protos = {}

       
        for label, proto_list in agg_protos.items():
            if len(proto_list) == 0:
                continue
            elif len(proto_list) == 1:
                global_classprotos[label] = {0: proto_list}
                global_protos[label] = proto_list[0].to(self.device)
                continue

            try:
                proto_matrix = torch.stack(proto_list).to(self.device)
            except Exception as e:
                print(f"[Error] Stacking failed for label {label}: {e}")
                continue

            distance_matrix = torch.cdist(proto_matrix, proto_matrix, p=2).cpu().numpy()
            np.fill_diagonal(distance_matrix, np.inf)  

            threshold = self.estimate_dynamic_threshold(distance_matrix, percentile=30)
     
            G = nx.Graph()
            G.add_nodes_from(range(len(proto_list)))

            for i in range(len(proto_list)):
                for j in range(i + 1, len(proto_list)):
                    if distance_matrix[i][j] < self.threshold:
                        G.add_edge(i, j)

            for i in range(len(proto_list))
                if G.degree[i] == 0:
                    j = np.argmin(distance_matrix[i])
                    G.add_edge(i, j)
                    #print(f"[Force-Link] Node {i} (label {label}) was isolated. Linked to node {j}.")

            global_classprotos[label] = {}
            subclass_centers = []

            for subclass_index, component in enumerate(nx.connected_components(G)):
                subclass_protos = [proto_list[i] for i in component]

              
                subclass_tensor = torch.stack(subclass_protos).to(self.device)
                subclass_center = torch.mean(subclass_tensor, dim=0)

             
                global_classprotos[label][subclass_index] = subclass_center
                subclass_centers.append(subclass_center)

            
            if len(subclass_centers) == 1:
                global_protos[label] = subclass_centers[0]
            else:
                global_protos[label] = torch.mean(torch.stack(subclass_centers), dim=0)

        return global_classprotos, global_protos

    def visualize_tsne_features(self, round_id):
       

        all_features, all_labels, all_client_ids = [], [], []

        for client in self.clients:
            testloader = client.load_test_data()
            model = load_item(client.role, 'model', client.save_folder_name)
            if model is None:
                continue
            model.eval()
            with torch.no_grad():
                for x, y in testloader:
                    x = x[0] if isinstance(x, list) else x
                    x = x.to(client.device)
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)  # 灰度转RGB
                    y = y.to(client.device)
                    rep = model.base(x).detach().cpu().numpy()
                    all_features.append(rep)
                    all_labels.extend(y.cpu().numpy())
                    all_client_ids.extend([client.id] * y.shape[0])

        if not all_features:
            print("[Error] No features extracted.")
            return

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        all_client_ids = np.array(all_client_ids)

        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        X_embedded = tsne.fit_transform(all_features)

        plt.figure(figsize=(8, 8))
        markers = ['o', 's', 'P', 'H', 'D', 'v', '^', '<', '>',
                   star_marker(4),  # 四角星
                   star_marker(6),  # 六角星
                   star_marker(7),  # 七角星
                   star_marker(8),
                   '8','*', 'X', 'd', 'h', 'p', star_marker(9),]
        class_palette = sns.color_palette("tab10", self.num_classes)

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        for cid in np.unique(all_client_ids):
            for cls in np.unique(all_labels):
                idx = (all_client_ids == cid) & (all_labels == cls)
                if np.sum(idx) == 0:
                    continue
                plt.scatter(
                    X_embedded[idx, 0], X_embedded[idx, 1],
                    marker=markers[cid % len(markers)],
                    c=[class_palette[cls]],
                    s=40,
                    edgecolors='none',
                    alpha=0.65
                )

        client_legend = [
            Line2D([0], [0], marker=markers[i % len(markers)],
                   color='black', linestyle='None', markersize=8,
                   label=f'{i}', markerfacecolor='gray')
            for i in range(len(set(all_client_ids)))
        ]

       
        class_legend = [
            Line2D([0], [0], marker='o', linestyle='None',
                   color=class_palette[i], label=f'{i}', markersize=8)
            for i in range(self.num_classes)
        ]

       
        legend1 = ax.legend(handles=client_legend, title="Client", loc='center left',
                            bbox_to_anchor=(1.01, 0.5), fontsize=9, title_fontsize=10)
        legend2 = ax.legend(handles=class_legend, title="Class", loc='center right',
                            bbox_to_anchor=(1.17, 0.5), fontsize=9, title_fontsize=10)
        ax.add_artist(legend1)

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in plt.gca().spines.values():
            spine.set_visible(True)  

        
        plt.tight_layout()
        plt.savefig(f"{round_id}_tsne.pdf", dpi=300)
        plt.show()

    def visualize_tsne_prototypes(self, round_id):
       
        all_vecs = []
        all_labels = []
        all_types = []
        for client in self.clients:
            client_protos = load_item(client.role, 'protos', client.save_folder_name)
            for label, vec_list in client_protos.items():
                for vec in vec_list:
                    all_vecs.append(vec.detach().cpu().numpy())
                    all_labels.append(label)
                    all_types.append("client")


        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        for label, vec in global_protos.items():
            all_vecs.append(vec.detach().cpu().numpy())
            all_labels.append(label)
            all_types.append("global")

        all_vecs = np.array(all_vecs)
        all_labels = np.array(all_labels)
        all_types = np.array(all_types)

        
        tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=1000)
        tsne_result = tsne.fit_transform(all_vecs)

        plt.figure(figsize=(7, 7))
        palette = sns.color_palette("hls", self.num_classes)

        for vec2d, label, typ in zip(tsne_result, all_labels, all_types):
            color = palette[label]
            if typ == "client":
                plt.scatter(vec2d[0], vec2d[1], s=125, marker='o',
                            facecolors=color, edgecolors='gray', linewidths=1, alpha=0.5)
            elif typ == "global":
                plt.scatter(vec2d[0], vec2d[1], s=130, marker='^',
                            facecolors=color, edgecolors='black', linewidths=1.5, alpha=0.9)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{round_id}_proto_tsne.pdf", dpi=300)
        plt.show()

    def visualize_tsne_prototypes2(self, round_id):

        all_vecs = []
        all_labels = []
        all_types = [] 

        for client in self.clients:
            client_protos = load_item(client.role, 'protos', client.save_folder_name)
            for label, vec_list in client_protos.items():
                for vec in vec_list:
                    all_vecs.append(vec.detach().cpu().numpy())
                    all_labels.append(label)
                    all_types.append("client")

        global_classprotos = load_item('Server', 'global_classprotos', self.save_folder_name)
        for label, subproto_dict in global_classprotos.items():
            for _, sub_proto in subproto_dict.items():
                all_vecs.append(sub_proto.detach().cpu().numpy())
                all_labels.append(label)
                all_types.append("subclass")

        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        for label, vec in global_protos.items():
            all_vecs.append(vec.detach().cpu().numpy())
            all_labels.append(label)
            all_types.append("global")

        all_vecs = np.array(all_vecs)
        all_labels = np.array(all_labels)
        all_types = np.array(all_types)

        tsne = TSNE(n_components=2, random_state=0, perplexity=5, n_iter=1000)
        tsne_result = tsne.fit_transform(all_vecs)

        plt.figure(figsize=(7, 7))
        palette = sns.color_palette("hls", self.num_classes)

        for vec2d, label, typ in zip(tsne_result, all_labels, all_types):
            color = palette[label]
            if typ == "client":
                plt.scatter(vec2d[0], vec2d[1], s=125, marker='o',
                            facecolors=color, edgecolors='gray', linewidths=1, alpha=0.5)
            elif typ == "subclass":
                plt.scatter(vec2d[0], vec2d[1], s=130, marker=star_marker(4),
                            facecolors=color, edgecolors='black', linewidths=1.2, alpha=0.8)
            elif typ == "global":
                plt.scatter(vec2d[0], vec2d[1], s=135, marker='^',
                            facecolors=color, edgecolors='black', linewidths=1.8, alpha=1.0)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{round_id}_proto_tsne_full.pdf", dpi=300)
        plt.show()

    def proto_aggregation(self, local_protos_list,i):
        agg_protos = defaultdict(list)

        for local_protos in local_protos_list:
            for label, proto_list in local_protos.items():
                agg_protos[label].extend(proto_list)
        """
        for local_protos in local_protos_list:
            for label in local_protos.keys():
                agg_protos[label].append(local_protos[label])
        """
        global_classprotos = {}
        global_protos={}

        for label, proto_list in agg_protos.items():
            if len(proto_list) == 0:
                continue

            if len(proto_list) == 1:
                global_classprotos[label] = {0: proto_list[0]}
                global_protos[label] = proto_list[0]
                continue

            try:
                proto_matrix = torch.stack(proto_list).to(self.device)


            except Exception as e:
                print(f"[Error] Stacking failed for label {label}: {e}")
                continue


            #distance_matrix = pairwise_distances(proto_matrix, metric='euclidean')

            distance_matrix = torch.cdist(proto_matrix, proto_matrix, p=2)

           
            distance_matrix = distance_matrix.cpu().numpy()
           # print('distance_matrix',distance_matrix)

            G_similar = nx.Graph()
            G_dissimilar = nx.Graph()
            N = len(proto_list)
            G_similar.add_nodes_from(range(N))
            G_dissimilar.add_nodes_from(range(N))


            for m in range(N):
                for n in range(m + 1, N):
                    if distance_matrix[m][n] < self.threshold:
                        G_similar.add_edge(m, n)
                    else:
                        G_dissimilar.add_edge(m, n)

            """
            if G.number_of_edges() == 0:
                print(f"[Fallback] No edges formed for label {label}, applying top-1 min-distance connection.")
                dist_matrix = distance_matrix.copy()
                np.fill_diagonal(dist_matrix, np.inf)
            
                for i in range(len(proto_list)):
                    j = np.argmin(dist_matrix[i])
                    if not np.isinf(dist_matrix[i][j]):
                        G.add_edge(i, j)
            """
            similar_subclasses = list(nx.connected_components(G_similar))
            dissimilar_subclasses = list(nx.connected_components(G_dissimilar))

            subclasses = {}
            similar_indices = set().union(*similar_subclasses)
            dissimilar_filtered = [
                [i for i in component if i not in similar_indices]
                for component in dissimilar_subclasses
            ]

         
            subclass0_cluster_protos = [
                torch.mean(torch.stack([proto_list[i] for i in component]), dim=0)
                for component in similar_subclasses if len(component) > 0
            ]
            subclasses[0] = subclass0_cluster_protos

        
            subclass1_cluster_protos = [
                torch.mean(torch.stack([proto_list[i] for i in component]), dim=0)
                for component in dissimilar_filtered if len(component) > 0
            ]
            subclasses[1] = subclass1_cluster_protos

            global_classprotos[label] = {}
            for idx in subclasses:
                if len(subclasses[idx]) > 0:
                    global_classprotos[label][idx] = torch.mean(torch.stack(subclasses[idx]), dim=0)

            all_subclass_protos = list(global_classprotos[label].values())
            global_protos[label] = torch.mean(torch.stack(all_subclass_protos), dim=0)

        return global_classprotos, global_protos




        """
            subclasses = {}
            subclasses[0] = [proto_list[i] for component in similar_subclasses for i in component]
            similar_indices = set().union(*similar_subclasses)

         
            subclasses[1] = [proto_list[i] for component in dissimilar_subclasses for i in component if
                             i not in similar_indices]

            global_classprotos[label] = {}
            for idx in subclasses:
                if len(subclasses[idx]) > 0:
                    global_classprotos[label][idx] = torch.mean(torch.stack(subclasses[idx]), dim=0)

            all_subclass_protos = list(global_classprotos[label].values())
            global_protos[label] = torch.mean(torch.stack(all_subclass_protos), dim=0)

        return global_classprotos, global_protos
        """
        """
            
            global_subclasses = {}
            for sub_index, protos in subclasses.items():
                if len(protos) > 0:
                    global_subclasses[sub_index] = torch.mean(torch.stack(protos), dim=0)

            all_subclass_protos = list(global_subclasses.values())
            if all_subclass_protos:
                class_global_proto = torch.mean(torch.stack(all_subclass_protos), dim=0)
            else:
                class_global_proto = None

        
            global_classprotos[label] = global_subclasses
            global_protos[label] = class_global_proto

        return global_classprotos, global_protos
        
        for label, proto_list in agg_protos.items():
            
            proto_matrix = torch.stack(proto_list).cpu().numpy()
            similarity_matrix = cosine_similarity(proto_matrix)
           
            G = nx.Graph()
            G.add_nodes_from(range(len(proto_matrix)))

            for i in range(len(proto_matrix)):
                for j in range(i + 1, len(proto_matrix)):
                    if similarity_matrix[i][j] > self.threshold:
                        G.add_edge(i, j)

            if G.number_of_edges() == 0:
                print(f"[Fallback] No edges formed for label {label}, applying top-1 similarity connection.")
                sim_matrix = similarity_matrix.copy()
                np.fill_diagonal(sim_matrix, -1)  
          
                for i in range(len(proto_list)):
                    j = np.argmax(sim_matrix[i])
                    if sim_matrix[i][j] > 0:  
                        G.add_edge(i, j)
        """

"""
   
def proto_aggregation(local_protos_list):
        agg_protos = defaultdict(list)
        for local_protos in local_protos_list:
            for label in local_protos.keys():
                agg_protos[label].append(local_protos[label])

        for [label, proto_list] in agg_protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos[label] = proto / len(proto_list)
            else:
                agg_protos[label] = proto_list[0].data

        return agg_protos
    """

def star_marker(num_points=4, inner_ratio=0.4):
    angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
    radius = np.tile([1, inner_ratio], num_points)
    verts = np.stack([np.cos(angles), np.sin(angles)]).T * radius[:, None]
    verts = np.append(verts, [verts[0]], axis=0)  # close shape
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return MarkerStyle(Path(verts, codes))


