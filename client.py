import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from sklearn.cluster import KMeans
import torch.nn.functional as F

class clientProtoexample(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = 0.2
        self.num_protos_per_class = 2
        self.temperature = 1
        self.alpha = 0.4

    def train(self):
        trainloader = self.load_train_data()
        # load client model and global protos
        model = load_item(self.role, 'model', self.save_folder_name)
        global_classprotos = load_item('Server', 'global_classprotos', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        
        protos = defaultdict(list)
        multiprotos = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)  
                output = model.head(rep)  
                loss_ls = self.loss(output, y)  
                
                loss_lr = self.calculate_contrastive_loss(rep, y, global_classprotos)
                loss_lkd = self.calculate_distillation_loss(rep, y, global_protos, global_classprotos)

                loss = loss_ls + self.lamda * loss_lr + (1 - self.lamda) * loss_lkd
                
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for y_c, reps in protos.items():
            reps = [rep.cpu().numpy() for rep in reps]
            if len(reps) < self.num_protos_per_class:
                
                avg_proto = np.mean(reps, axis=0)
                multiprotos[y_c].append(torch.tensor(avg_proto, dtype=torch.float32).to(self.device))
            else:
                
                kmeans = KMeans(n_clusters=self.num_protos_per_class, random_state=0)
                kmeans.fit(reps)
                for center in kmeans.cluster_centers_:
                    multiprotos[y_c].append(torch.tensor(center, dtype=torch.float32).to(self.device))

        save_item(multiprotos, self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_classprotos = load_item('Server', 'global_classprotos', self.save_folder_name)
        model.eval()

        test_acc = 0
        test_num = 0
        
        if global_classprotos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)
                    
                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    
                    """
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)
                    """

                    for i, r in enumerate(rep):  
                        for class_id, sub_proto_dict in global_classprotos.items():
                            min_dist = float('inf')
                            for sub_id, proto in sub_proto_dict.items():
                                dist = self.loss_mse(r, proto)
                                if dist < min_dist:
                                    min_dist = dist
                            output[i, class_id] = min_dist
                            
                    """
                    for i, r in enumerate(rep):  
                        for class_id in range(self.num_classes):
                            min_subclass_dist = float('inf')
                           
                            if class_id in global_classprotos:
                                for sub_id, proto in global_classprotos[class_id].items():
                                    dist = self.loss_mse(r, proto)
                                    if dist < min_subclass_dist:
                                        min_subclass_dist = dist
                            else:
                                min_subclass_dist = float('inf')

                           
                            if class_id in global_protos:
                                global_proto_dist = self.loss_mse(r, global_protos[class_id])
                            else:
                                global_proto_dist = float('inf')

                            
                            output[i, class_id] = min(min_subclass_dist, global_proto_dist)
                    """
                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    
    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_classprotos = load_item('Server', 'global_classprotos', self.save_folder_name)  
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)  
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                output = model.head(rep)
                loss_ls = self.loss(output, y)

                loss_lr = self.calculate_contrastive_loss(rep, y, global_classprotos)

                # 3. 计算蒸馏损失 L_KD
                loss_lkd = self.calculate_distillation_loss(rep, y, global_protos, global_classprotos)

                # 4. 综合损失
                total_loss = loss_ls + self.lamda * loss_lr + (1 - self.lamda) * loss_lkd
                train_num += y.shape[0]
                losses += total_loss.item() * y.shape[0]

        return losses, train_num
    """
    if global_protos is not None:
        proto_new = copy.deepcopy(rep.detach())  
        for i, yy in enumerate(y):
            y_c = yy.item()
            if type(global_protos[y_c]) != type([]):
                proto_new[i, :] = global_protos[y_c].data
        loss += self.loss_mse(proto_new, rep) * self.lamda  # LR
    
    proto_new = copy.deepcopy(rep.detach())
    for i, yy in enumerate(y):
        y_c = yy.item()
        if type(global_classprotos[y_c]) != type([]):
            proto_new[i, :] = global_classprotos[y_c].data
    # 正对比对 ψps
    pos_proto = None
    max_sim = -float('inf')
    sim = F.cosine_similarity(proto_new, rep)
    """
    def calculate_contrastive_loss(self, rep, y, global_classprotos):
        loss = 0.0

        
        if global_classprotos is not None:
            for i, feature in enumerate(rep):
                label = y[i].item()

                subclass_protos = global_classprotos[label]  #

                
                pos_proto = None
                max_sim = -float('inf')
                for sub_index, proto in subclass_protos.items():
                    sim = F.cosine_similarity(feature.unsqueeze(0), proto.unsqueeze(0)).item()
                    if sim > max_sim:
                        max_sim = sim
                        pos_proto = proto

                psi_ps = torch.exp(torch.tensor(max_sim/self.temperature, device=self.device))

                psi_psp = 0.0
                for sub_index, proto in subclass_protos.items():
                    if not torch.equal(proto, pos_proto):  
                        sim = F.cosine_similarity(feature.unsqueeze(0), proto.unsqueeze(0)).item()
                        psi_psp += torch.exp(torch.tensor(sim/self.temperature, device=self.device))

             
                psi_np = 0.0
                for other_label, other_subclass_protos in global_classprotos.items():
                    if other_label == label:
                        continue
                    for sub_index, proto in other_subclass_protos.items():
                        sim = F.cosine_similarity(feature.unsqueeze(0), proto.unsqueeze(0)).item()
                        psi_np += torch.exp(torch.tensor(sim/self.temperature, device=self.device))

                loss += -torch.log(psi_ps / (psi_ps + self.alpha * psi_psp + (1 - self.alpha) * psi_np))
        loss = loss / rep.shape[0]
        return loss
    """
    def calculate_distillation_loss(self, rep, y, global_protos, global_classprotos):
        loss = 0.0
        if global_classprotos and global_protos is not None:
            for i, feature in enumerate(rep):
                label = y[i].item()

                global_proto = global_protos[label] 
                subclass_protos = global_classprotos[label]  

                
                for sub_index, subclass_proto in subclass_protos.items():
                    sim_global = F.cosine_similarity(global_proto.unsqueeze(0), subclass_proto.unsqueeze(0))
                    sim_local = F.cosine_similarity(feature.unsqueeze(0), global_proto.unsqueeze(0))
                    loss += (sim_global - sim_local) ** 2

        return loss / rep.shape[0]
    """

    def calculate_distillation_loss(self, rep, y, global_protos, global_classprotos):
           
            loss=0.0
            proto_align_loss = 0.0
            feature_align_loss = 0.0
            subproto_count = 0
            if global_classprotos and global_protos is not None:
                for c in global_protos:
                    pE = global_protos[c]  
                    sub_protos = global_classprotos[c]  
                    for r, sub_p in sub_protos.items():
                        proto_align_loss += F.mse_loss(pE, sub_p, reduction='mean')
                        subproto_count += 1

                for i, feat in enumerate(rep):
                    label = y[i].item()
                    pE = global_protos[label]
                    feature_align_loss += F.mse_loss(feat, pE, reduction='mean')

                
                if subproto_count == 0:
                    return feature_align_loss / rep.shape[0]

                loss = proto_align_loss / subproto_count + feature_align_loss / rep.shape[0]
            return loss

def agg_func(protos):
  
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos



