import pgl
import paddle.fluid as fluid
import paddle.fluid.layers as L
import pgl.layers.conv as conv
from pgl.utils import paddle_helper

def get_norm(indegree):
    float_degree = L.cast(indegree, dtype="float32")
    float_degree = L.clamp(float_degree, min=1.0)
    norm = L.pow(float_degree, factor=-0.5) 
    return norm
    

class GCN(object):
    """Implement of GCN
    """
    def __init__(self,config,num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        
        for i in range(self.num_layers):

            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]


            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)

            feature = L.dropout(
                    feature,
                    self.dropout,
                    dropout_implementation='upscale_in_train')

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
                     name="output")

        return feature

class ResGCN(object):
    """Implement of ResGCN
    """
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        feature = L.fc(feature, size=self.hidden_size, name="init_feature")
        
        for i in range(self.num_layers):

            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]

            res_feature = feature
            feature = pgl.layers.gcn(ngw,
                feature,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % i)
            feature = res_feature + feature 
            # feature = L.dropout(
            #         feature,
            #         self.dropout,
            #         dropout_implementation='upscale_in_train')
            feature = L.relu(feature)
            feature = L.layer_norm(feature, name="ln_%s" % i)

        if phase == "train": 
            ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
            norm = get_norm(ngw.indegree())
        else:
            ngw = graph_wrapper
            norm = graph_wrapper.node_feat["norm"]

        feature = conv.gcn(ngw,
                     feature,
                     self.num_class,
                     activation=None,
                     norm=norm,
                     name="output")

        return feature

class GAT(object):
    """Implement of GAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
                
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation="elu",
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature


class ResGAT(object):
    """Implement of ResGAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        # feature [num_nodes, 100]
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, size=self.hidden_size * self.num_heads, name="init_feature")
        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            
            res_feature = feature
            # res_feature [num_nodes, hidden_size * n_heads]
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation=None,
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)
            # feature [num_nodes, num_heads * hidden_size]
            feature = res_feature + feature 
            # [num_nodes, num_heads * hidden_size] + [ num_nodes, hidden_size * n_heads]
            feature = L.relu(feature)
            feature = L.layer_norm(feature, name="ln_%s" % i)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=1,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        return feature


class ResGATII(object):
    """Implement of ResGAT"""
    def __init__(self, config, num_class):
        self.num_class = num_class 
        self.num_layers = config.get("num_layers", 1)
        self.num_heads = config.get("num_heads", 8)
        self.hidden_size = config.get("hidden_size", 8)
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        # feature [num_nodes, 100]
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0
        feature = L.fc(feature, size=self.hidden_size * self.num_heads, name="init_feature")
        for i in range(self.num_layers):
            ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
            
            res_feature = feature
            # res_feature [num_nodes, hidden_size * n_heads]
            feature = conv.gat(ngw,
                                feature,
                                self.hidden_size,
                                activation=None,
                                name="gat_layer_%s" % i,
                                num_heads=self.num_heads,
                                feat_drop=self.feat_dropout,
                                attn_drop=self.attn_dropout)
            # feature [num_nodes, num_heads * hidden_size]
            feature = res_feature + feature 
            # [num_nodes, num_heads * hidden_size] + [ num_nodes, hidden_size * n_heads]
            feature = L.relu(feature)
            feature = L.layer_norm(feature, name="ln_%s" % i)

        ngw = pgl.sample.edge_drop(graph_wrapper, edge_dropout) 
        feature = conv.gat(ngw,
                     feature,
                     self.num_class,
                     num_heads=8,
                     activation=None,
                     feat_drop=self.feat_dropout,
                     attn_drop=self.attn_dropout,
                     name="output")
        # feature [num_nodes, num_heads * num_class]
        feature = L.relu(feature)
        feature = L.fc(feature, size=self.num_class, name="output_feature")
        return feature

   
class APPNP(object):
    """Implement of APPNP"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout = config.get("dropout", 0.5)
        self.alpha = config.get("alpha", 0.1)
        self.k_hop = config.get("k_hop", 10)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)

        feature = L.dropout(
            feature,
            self.dropout,
            dropout_implementation='upscale_in_train')

        feature = L.fc(feature, self.num_class, act=None, name="output")

        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=edge_dropout,
            alpha=self.alpha,
            k_hop=self.k_hop)
        return feature

class SGC(object):
    """Implement of SGC"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)

    def forward(self, graph_wrapper, feature, phase):
        feature = conv.appnp(graph_wrapper,
            feature=feature,
            edge_dropout=0,
            alpha=0,
            k_hop=self.num_layers)
        feature.stop_gradient=True
        feature = L.fc(feature, self.num_class, act=None, bias_attr=False, name="output")
        return feature

 
class GCNII(object):
    """Implement of GCNII"""
    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 32)
        self.dropout = config.get("dropout", 0.6)
        self.alpha = config.get("alpha", 0.1)
        self.lambda_l = config.get("lambda_l", 0.5)
        self.k_hop = config.get("k_hop", 32)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        if phase == "train": 
            edge_dropout = self.edge_dropout
        else:
            edge_dropout = 0

        for i in range(self.num_layers):
            feature = L.fc(feature, self.hidden_size, act="relu", name="lin%s" % i)
            feature = L.dropout(
                feature,
                self.dropout,
                dropout_implementation='upscale_in_train')

        feature = conv.gcnii(graph_wrapper,
            feature=feature,
            name="gcnii",
            activation="relu",
            lambda_l=self.lambda_l,
            alpha=self.alpha,
            dropout=self.dropout,
            k_hop=self.k_hop)

        feature = L.fc(feature, self.num_class, act=None, name="output")
        return feature

class DeeperGCN(object):
    """Implementation of DeeperGCN, see the paper
    "DeeperGCN: All You Need to Train Deeper GCNs" in
    https://arxiv.org/pdf/2006.07739.pdf

    Args:
        gw: Graph wrapper object

        feature: A tensor with shape (num_nodes, feature_size)

        num_layers: num of layers in DeeperGCN

        hidden_size: hidden_size in DeeperGCN

        num_tasks: final prediction
        
        name: deeper gcn layer names

        dropout_prob: dropout prob in DeeperGCN

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """

    def __init__(self, config, num_class):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size = config.get("hidden_size", 64)
        self.dropout_prob = config.get("dropout", 0.5)
        self.num_tasks=num_class
        self.num_heads = config.get("num_heads", 8)
        self.name="deepercnn"
        self.feat_dropout = config.get("feat_drop", 0.6)
        self.attn_dropout = config.get("attn_drop", 0.6)
        self.edge_dropout = config.get("edge_dropout", 0.0)

    def forward(self, graph_wrapper, feature, phase):
        beta = "dynamic"
        if phase == "train": 
            self.dropout_prob = self.dropout_prob
        else:
            self.dropout_prob = 0
        gw=graph_wrapper
        feature = fluid.layers.fc(feature,
                     self.hidden_size,
                     bias_attr=False,
                     param_attr=fluid.ParamAttr(name=self.name + '_weight'))
    
        output = pgl.layers.gen_conv(gw, feature, name=self.name+"_gen_conv_0", beta=beta)

        for layer in range(self.num_layers):
        # LN/BN->ReLU->GraphConv->Res
            old_output = output
            if phase == "train":
                ngw = pgl.sample.edge_drop(graph_wrapper, self.edge_dropout) 
                norm = get_norm(ngw.indegree())
            else:
                ngw = graph_wrapper
                norm = graph_wrapper.node_feat["norm"]

            
        # 1. Layer Norm
            output = fluid.layers.layer_norm(
                output,
                begin_norm_axis=1,
                param_attr=fluid.ParamAttr(
                    name="norm_scale_%s_%d" % (self.name, layer),
                    initializer=fluid.initializer.Constant(1.0)),
                bias_attr=fluid.ParamAttr(
                    name="norm_bias_%s_%d" % (self.name, layer),
                    initializer=fluid.initializer.Constant(0.0)))

        # 2. ReLU
            output = fluid.layers.relu(output)

        #3. dropout
            output = fluid.layers.dropout(output, 
                dropout_prob=self.dropout_prob,
                dropout_implementation="upscale_in_train")

        #4 gen_conv
            
            
            
            output = pgl.layers.gen_conv(gw, output, 
                name=self.name+"_gen_conv_%d"%layer, beta=beta)
            

        #5 res
            output = output + old_output
            #output = (output + old_output + gat_out+gcn_out)/4
            #output = (output + gat_out + gcn_out)/3

    # final layer: LN + relu + droput
    # final prediction
        gat_out = conv.gat(ngw,
                            output,
                            self.hidden_size,
                            activation="elu",
                            name="gat_layer_%s" % self.num_layers,
                            num_heads=1,
                            feat_drop=self.feat_dropout,
                            attn_drop=self.attn_dropout)
            
        gcn_out = pgl.layers.gcn(ngw,
                output,
                self.hidden_size,
                activation="relu",
                norm=norm,
                name="layer_%s" % self.num_layers)
        
        ouput=output+gat_out+gcn_out

        output = fluid.layers.layer_norm(
                output,
                begin_norm_axis=1,
                param_attr=fluid.ParamAttr(
                    name="norm_scale_%s_%d" % (self.name, self.num_layers),
                    initializer=fluid.initializer.Constant(1.0)),
                bias_attr=fluid.ParamAttr(
                    name="norm_bias_%s_%d" % (self.name, self.num_layers),
                    initializer=fluid.initializer.Constant(0.0)))
        output = fluid.layers.relu(output)
        output = fluid.layers.dropout(output, 
                dropout_prob=self.dropout_prob,
                dropout_implementation="upscale_in_train")

        output = fluid.layers.fc(output,
                    self.num_tasks,
                    bias_attr=False,
                    param_attr=fluid.ParamAttr(name=self.name + '_final_weight'))


        return output


class GaAN(object):
    def __init__(self,
                 config,
                 num_class,
                 num_layers=1,
                 hidden_size_a=24,
                 hidden_size_v=32,
                 hidden_size_m=64,
                 hidden_size_o=128,
                 heads=8,
                 act='relu',
                 name="GaAN"):
        self.num_class = num_class
        self.num_layers = config.get("num_layers", 1)
        self.hidden_size_a = 48
        self.hidden_size_v = 64
        self.hidden_size_m = 128
        self.hidden_size_o = 256
        self.act = act
        self.name = name
        self.heads = heads
        self.drop_rate=config.get("dropout", 0.5)

    def GaANConv(self, gw, feature, name):
        feat_key = fluid.layers.fc(
            feature,
            self.hidden_size_a * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_key'))
        # N * (D2 * M)
        feat_value = fluid.layers.fc(
            feature,
            self.hidden_size_v * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_value'))
        # N * (D1 * M)
        feat_query = fluid.layers.fc(
            feature,
            self.hidden_size_a * self.heads,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_query'))
        # N * Dm
        feat_gate = fluid.layers.fc(
            feature,
            self.hidden_size_m,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_gate'))

        # send
        message = gw.send(
            self.send_func,
            nfeat_list=[('node_feat', feature), ('feat_key', feat_key),
                        ('feat_value', feat_value), ('feat_query', feat_query),
                        ('feat_gate', feat_gate)],
            efeat_list=None, )

        # recv
        output = gw.recv(message, self.recv_func)
        output = fluid.layers.fc(
            output,
            self.hidden_size_o,
            bias_attr=False,
            param_attr=fluid.ParamAttr(name=name + '_project_output'))
        output = fluid.layers.leaky_relu(output, alpha=0.1)
        output = fluid.layers.dropout(output, dropout_prob=0.1)
        return output

    def forward(self, gw, feature,phase):
        if phase == "train": 
            self.drop_rate = self.drop_rate
        else:
            self.drop_rate = 0
        for i in range(self.num_layers):
            feature = self.GaANConv(gw, feature, self.name + '_' + str(i))
            feature = fluid.layers.dropout(feature, dropout_prob=self.drop_rate)
            feature = L.fc(feature, self.num_class, act=None, name="output")
        return feature

    def send_func(self, src_feat, dst_feat, edge_feat):
        # E * (M * D1)
        feat_query, feat_key = dst_feat['feat_query'], src_feat['feat_key']
        # E * M * D1
        old = feat_query
        feat_query = fluid.layers.reshape(
            feat_query, [-1, self.heads, self.hidden_size_a])
        feat_key = fluid.layers.reshape(feat_key,
                                        [-1, self.heads, self.hidden_size_a])
        # E * M
        alpha = fluid.layers.reduce_sum(feat_key * feat_query, dim=-1)

        return {
            'dst_node_feat': dst_feat['node_feat'],
            'src_node_feat': src_feat['node_feat'],
            'feat_value': src_feat['feat_value'],
            'alpha': alpha,
            'feat_gate': src_feat['feat_gate']
        }

    def recv_func(self, message):
        dst_feat = message['dst_node_feat']
        src_feat = message['src_node_feat']
        x = fluid.layers.sequence_pool(dst_feat, 'average')
        z = fluid.layers.sequence_pool(src_feat, 'average')

        feat_gate = message['feat_gate']
        g_max = fluid.layers.sequence_pool(feat_gate, 'max')

        g = fluid.layers.concat([x, g_max, z], axis=1)
        g = fluid.layers.fc(g, self.heads, bias_attr=False, act="sigmoid")

        # softmax
        alpha = message['alpha']
        alpha = paddle_helper.sequence_softmax(alpha)  # E * M

        feat_value = message['feat_value']  # E * (M * D2)
        old = feat_value
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads, self.hidden_size_v])  # E * M * D2
        feat_value = fluid.layers.elementwise_mul(feat_value, alpha, axis=0)
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads * self.hidden_size_v])  # E * (M * D2)
        feat_value = fluid.layers.lod_reset(feat_value, old)

        feat_value = fluid.layers.sequence_pool(feat_value,
                                                'sum')  # N * (M * D2)
        feat_value = fluid.layers.reshape(
            feat_value, [-1, self.heads, self.hidden_size_v])  # N * M * D2
        output = fluid.layers.elementwise_mul(feat_value, g, axis=0)
        output = fluid.layers.reshape(
            output, [-1, self.heads * self.hidden_size_v])  # N * (M * D2)
        output = fluid.layers.concat([x, output], axis=1)

        return output

